//! Protocol conversion for chat-template models: OpenAI Chat Completions,
//! OpenAI Responses and Anthropic Messages requests become one neutral chat
//! request (OpenAI-style messages for the model's chat template, the tools,
//! the sampling options and the response settings), the way vLLM and SGLang
//! prepare requests for Hugging Face chat templates.
//!
//! Media blocks were replaced by numbered synthetic image blocks before
//! conversion (`media::extract`); they become [`Part::Media`] here.

use serde_json::{Map, Value, json};

use crate::media;
use crate::protocol::{PROTOCOL_CHAT_COMPLETIONS, PROTOCOL_MESSAGES, PROTOCOL_RESPONSES};

/// One part of a message's content.
#[derive(Debug, Clone, PartialEq)]
pub enum Part {
    Text(String),
    /// An image or video: index into the request's extracted media.
    Media(usize),
}

/// Message content: a plain string, or parts.
#[derive(Debug, Clone, PartialEq)]
pub enum Content {
    Text(String),
    Parts(Vec<Part>),
}

impl Content {
    pub fn empty() -> Self {
        Self::Text(String::new())
    }

    /// The text of the content, parts joined with newlines (vLLM's string
    /// content format); `None` when it holds media.
    pub fn joined_text(&self) -> Option<String> {
        match self {
            Self::Text(s) => Some(s.clone()),
            Self::Parts(parts) => {
                let mut texts = Vec::with_capacity(parts.len());
                for p in parts {
                    match p {
                        Part::Text(t) => texts.push(t.as_str()),
                        Part::Media(_) => return None,
                    }
                }
                Some(texts.join("\n"))
            }
        }
    }
}

/// A historical tool call.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    /// The arguments: a JSON object when they parse as one (as vLLM passes
    /// them to templates), else the string as sent.
    pub arguments: Value,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Message {
    /// `system`, `user`, `assistant` or `tool`.
    pub role: &'static str,
    pub content: Content,
    pub reasoning: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub tool_call_id: Option<String>,
    pub name: Option<String>,
}

impl Message {
    fn new(role: &'static str, content: Content) -> Self {
        Self {
            role,
            content,
            reasoning: None,
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ToolChoice {
    Auto,
    None,
    Required,
    /// Call this function (`tools` already holds only it).
    Function(String),
}

impl ToolChoice {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::None => "none",
            Self::Required | Self::Function(_) => "required",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ResponseFormat {
    Text,
    JsonObject,
    JsonSchema(Value),
    Regex(String),
}

impl ResponseFormat {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::JsonObject => "json_object",
            Self::JsonSchema(_) => "json_schema",
            Self::Regex(_) => "regex",
        }
    }
}

/// A request in protocol-neutral form.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatRequest {
    pub messages: Vec<Message>,
    /// OpenAI function tools (`{"type": "function", "function": {...}}`).
    pub tools: Vec<Value>,
    pub tool_choice: ToolChoice,
    /// Explicit thinking choice of the request.
    pub thinking: Option<bool>,
    pub reasoning_effort: Option<String>,
    pub response_format: ResponseFormat,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<i64>,
    pub min_p: Option<f64>,
    pub presence_penalty: Option<f64>,
    pub frequency_penalty: Option<f64>,
    pub repetition_penalty: Option<f64>,
    pub seed: Option<u64>,
    pub stop: Vec<String>,
    pub logprobs: bool,
    pub top_logprobs: Option<u8>,
    pub stream: bool,
    pub include_usage: bool,
    /// Extra chat template variables (vLLM `chat_template_kwargs`).
    pub template_kwargs: Map<String, Value>,
    /// Continue the final assistant message instead of opening a new turn.
    pub continue_final: bool,
}

impl ChatRequest {
    fn new() -> Self {
        Self {
            messages: Vec::new(),
            tools: Vec::new(),
            tool_choice: ToolChoice::Auto,
            thinking: None,
            reasoning_effort: None,
            response_format: ResponseFormat::Text,
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            min_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            repetition_penalty: None,
            seed: None,
            stop: Vec::new(),
            logprobs: false,
            top_logprobs: None,
            stream: false,
            include_usage: false,
            template_kwargs: Map::new(),
            continue_final: false,
        }
    }
}

type Res<T> = Result<T, String>;

/// Convert a request body of `protocol`.
pub fn convert(protocol: &str, body: &Value) -> Res<ChatRequest> {
    let obj = body
        .as_object()
        .ok_or("the request body must be a JSON object")?;
    let mut req = match protocol {
        PROTOCOL_CHAT_COMPLETIONS => chat_completions(obj)?,
        PROTOCOL_MESSAGES => messages(obj)?,
        PROTOCOL_RESPONSES => responses(obj)?,
        other => {
            return Err(format!(
                "unknown protocol {other:?} (want chat_completions, responses or messages)"
            ));
        }
    };
    if req.messages.is_empty() {
        return Err("the request has no messages".into());
    }
    if req.continue_final {
        match req.messages.last() {
            Some(m) if m.role == "assistant" => {}
            _ => return Err("continuing the final message needs a final assistant message".into()),
        }
    }
    resolve_tool_choice(&mut req)?;
    validate_sampling(&req)?;
    Ok(req)
}

// ---------------------------------------------------------------------------
// Field helpers
// ---------------------------------------------------------------------------

fn opt_str<'a>(o: &'a Map<String, Value>, k: &str) -> Res<Option<&'a str>> {
    match o.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s)),
        Some(_) => Err(format!("{k} must be a string")),
    }
}

fn opt_bool(o: &Map<String, Value>, k: &str) -> Res<Option<bool>> {
    match o.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Bool(b)) => Ok(Some(*b)),
        Some(_) => Err(format!("{k} must be a boolean")),
    }
}

fn opt_f64(o: &Map<String, Value>, k: &str) -> Res<Option<f64>> {
    match o.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v.as_f64().map(Some).ok_or(format!("{k} must be a number")),
    }
}

fn opt_i64(o: &Map<String, Value>, k: &str) -> Res<Option<i64>> {
    match o.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_i64()
            .map(Some)
            .ok_or(format!("{k} must be an integer")),
    }
}

fn opt_u32(o: &Map<String, Value>, k: &str) -> Res<Option<u32>> {
    match opt_i64(o, k)? {
        None => Ok(None),
        Some(n) if n >= 1 && n <= u32::MAX as i64 => Ok(Some(n as u32)),
        Some(n) => Err(format!("{k} must be at least 1, got {n}")),
    }
}

fn opt_obj<'a>(o: &'a Map<String, Value>, k: &str) -> Res<Option<&'a Map<String, Value>>> {
    match o.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Object(m)) => Ok(Some(m)),
        Some(_) => Err(format!("{k} must be an object")),
    }
}

fn opt_array<'a>(o: &'a Map<String, Value>, k: &str) -> Res<Option<&'a Vec<Value>>> {
    match o.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Array(a)) => Ok(Some(a)),
        Some(_) => Err(format!("{k} must be an array")),
    }
}

fn as_obj<'a>(v: &'a Value, at: &str) -> Res<&'a Map<String, Value>> {
    v.as_object()
        .ok_or_else(|| format!("{at} must be an object"))
}

/// `stop`: a string or a list of strings.
fn stop_list(v: Option<&Value>, key: &str) -> Res<Vec<String>> {
    let list = match v {
        None | Some(Value::Null) => return Ok(Vec::new()),
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a
            .iter()
            .map(|x| {
                x.as_str()
                    .map(str::to_string)
                    .ok_or(format!("{key} must hold strings"))
            })
            .collect::<Res<Vec<_>>>()?,
        Some(_) => return Err(format!("{key} must be a string or a list of strings")),
    };
    if list.iter().any(String::is_empty) {
        return Err(format!("{key} must not hold empty strings"));
    }
    if list.len() > 16 {
        return Err(format!("{key} holds {} sequences, at most 16", list.len()));
    }
    Ok(list)
}

/// A media part of a synthetic block, if `block` is one.
fn synthetic_media(block: &Map<String, Value>) -> Option<usize> {
    let url = match block.get("type").and_then(Value::as_str)? {
        "image_url" => match block.get("image_url")? {
            Value::Object(o) => o.get("url")?.as_str()?,
            Value::String(s) => s.as_str(),
            _ => return None,
        },
        "image" => block.get("source")?.get("url")?.as_str()?,
        "input_image" => block.get("image_url")?.as_str()?,
        _ => return None,
    };
    media::synthetic_index(url)
}

/// Parse tool-call arguments: an object when the string holds one.
fn parse_arguments(args: &str) -> Value {
    match serde_json::from_str::<Value>(args) {
        Ok(v @ Value::Object(_)) => v,
        _ => Value::String(args.to_string()),
    }
}

fn function_tool(name: &str, description: Option<&Value>, parameters: Option<&Value>) -> Value {
    let mut f = Map::new();
    f.insert("name".into(), json!(name));
    if let Some(d) = description.filter(|d| !d.is_null()) {
        f.insert("description".into(), d.clone());
    }
    if let Some(p) = parameters.filter(|p| !p.is_null()) {
        f.insert("parameters".into(), p.clone());
    }
    json!({"type": "function", "function": f})
}

fn validate_tool_name(name: &str, at: &str) -> Res<()> {
    if name.is_empty() || name.len() > 128 {
        return Err(format!("{at}: a tool name needs 1 to 128 characters"));
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | ':'))
    {
        return Err(format!(
            "{at}: tool name {name:?} may only hold letters, digits, '_', '-', '.' and ':'"
        ));
    }
    Ok(())
}

fn tool_name(tool: &Value) -> &str {
    tool.get("function")
        .and_then(|f| f.get("name"))
        .and_then(Value::as_str)
        .unwrap_or("")
}

/// Thinking from `thinking: {"type": ...}` (Anthropic, DeepSeek API).
fn thinking_type(o: &Map<String, Value>) -> Res<Option<bool>> {
    let Some(t) = opt_obj(o, "thinking")? else {
        return Ok(None);
    };
    match t.get("type").and_then(Value::as_str) {
        Some("enabled" | "adaptive") => Ok(Some(true)),
        Some("disabled") => Ok(Some(false)),
        _ => Err("thinking.type must be enabled, adaptive or disabled".into()),
    }
}

fn effort_thinking(effort: Option<&str>) -> Option<bool> {
    effort.map(|e| e != "none")
}

// ---------------------------------------------------------------------------
// OpenAI Chat Completions
// ---------------------------------------------------------------------------

fn chat_completions(o: &Map<String, Value>) -> Res<ChatRequest> {
    let mut req = ChatRequest::new();
    let msgs = opt_array(o, "messages")?.ok_or("messages is required")?;
    for (i, m) in msgs.iter().enumerate() {
        let at = format!("messages.{i}");
        let m = as_obj(m, &at)?;
        let role = opt_str(m, "role")?.ok_or(format!("{at}.role is required"))?;
        let content = |allow_media: bool| chat_content(m.get("content"), &at, allow_media);
        let msg = match role {
            "system" | "developer" => Message::new("system", content(false)?),
            "user" => Message::new("user", content(true)?),
            "assistant" => {
                let mut msg = Message::new("assistant", content(false)?);
                msg.reasoning = match opt_str(m, "reasoning_content")? {
                    Some(r) => Some(r.to_string()),
                    None => opt_str(m, "reasoning")?.map(str::to_string),
                };
                for (j, tc) in opt_array(m, "tool_calls")?
                    .into_iter()
                    .flatten()
                    .enumerate()
                {
                    let at = format!("{at}.tool_calls.{j}");
                    let tc = as_obj(tc, &at)?;
                    let f = opt_obj(tc, "function")?.ok_or(format!("{at}.function is required"))?;
                    let name =
                        opt_str(f, "name")?.ok_or(format!("{at}.function.name is required"))?;
                    let arguments = match f.get("arguments") {
                        Some(Value::String(s)) => parse_arguments(s),
                        Some(v @ Value::Object(_)) => v.clone(),
                        None | Some(Value::Null) => json!({}),
                        Some(_) => return Err(format!("{at}.function.arguments must be a string")),
                    };
                    msg.tool_calls.push(ToolCall {
                        id: opt_str(tc, "id")?.unwrap_or_default().to_string(),
                        name: name.to_string(),
                        arguments,
                    });
                }
                if opt_bool(m, "prefix")?.unwrap_or(false) {
                    if i + 1 != msgs.len() {
                        return Err(format!("{at}: only the final message can be a prefix"));
                    }
                    req.continue_final = true;
                }
                msg
            }
            "tool" => {
                let mut msg = Message::new("tool", content(true)?);
                msg.tool_call_id = opt_str(m, "tool_call_id")?.map(str::to_string);
                msg.name = opt_str(m, "name")?.map(str::to_string);
                msg
            }
            other => {
                return Err(format!(
                    "{at}.role {other:?} is not supported (want system, developer, user, assistant or tool)"
                ));
            }
        };
        req.messages.push(msg);
    }

    for (i, t) in opt_array(o, "tools")?.into_iter().flatten().enumerate() {
        let at = format!("tools.{i}");
        let to = as_obj(t, &at)?;
        if opt_str(to, "type")?.unwrap_or("function") != "function" {
            return Err(format!("{at}: only function tools are supported"));
        }
        let f = opt_obj(to, "function")?.ok_or(format!("{at}.function is required"))?;
        let name = opt_str(f, "name")?.ok_or(format!("{at}.function.name is required"))?;
        validate_tool_name(name, &format!("{at}.function"))?;
        req.tools.push(t.clone());
    }
    req.tool_choice = match o.get("tool_choice") {
        None | Some(Value::Null) => ToolChoice::Auto,
        Some(Value::String(s)) => named_choice(s)?,
        Some(Value::Object(c)) => {
            let name = c
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(Value::as_str)
                .ok_or("tool_choice.function.name is required")?;
            ToolChoice::Function(name.to_string())
        }
        Some(_) => return Err("tool_choice must be a string or an object".into()),
    };

    req.response_format = match opt_obj(o, "response_format")? {
        None => ResponseFormat::Text,
        Some(f) => match opt_str(f, "type")? {
            Some("text") | None => ResponseFormat::Text,
            Some("json_object") => ResponseFormat::JsonObject,
            Some("json_schema") => {
                let js =
                    opt_obj(f, "json_schema")?.ok_or("response_format.json_schema is required")?;
                json_schema(js.get("schema"), "response_format.json_schema.schema")?
            }
            Some("regex") => ResponseFormat::Regex(
                opt_str(f, "regex")?
                    .ok_or("response_format.regex is required")?
                    .to_string(),
            ),
            Some(other) => return Err(format!("response_format.type {other:?} is not supported")),
        },
    };

    req.max_tokens = match opt_u32(o, "max_completion_tokens")? {
        Some(n) => Some(n),
        None => opt_u32(o, "max_tokens")?,
    };
    sampling(o, &mut req)?;
    if let Some(n) = opt_i64(o, "n")?
        && n != 1
    {
        return Err(format!("n must be 1, got {n}"));
    }
    req.stop = stop_list(o.get("stop"), "stop")?;
    req.logprobs = opt_bool(o, "logprobs")?.unwrap_or(false);
    if let Some(n) = opt_i64(o, "top_logprobs")? {
        if !(0..=20).contains(&n) {
            return Err(format!("top_logprobs must be in [0, 20], got {n}"));
        }
        if !req.logprobs {
            return Err("top_logprobs requires logprobs: true".into());
        }
        req.top_logprobs = Some(n as u8);
    }
    req.stream = opt_bool(o, "stream")?.unwrap_or(false);
    req.include_usage = opt_obj(o, "stream_options")?
        .and_then(|s| s.get("include_usage"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if let Some(kw) = opt_obj(o, "chat_template_kwargs")? {
        req.template_kwargs = kw.clone();
    }
    req.reasoning_effort = opt_str(o, "reasoning_effort")?.map(str::to_string);
    req.thinking = match thinking_type(o)? {
        Some(t) => Some(t),
        None => match req.template_kwargs.get("enable_thinking") {
            Some(Value::Bool(b)) => Some(*b),
            _ => effort_thinking(req.reasoning_effort.as_deref()),
        },
    };
    if opt_bool(o, "continue_final_message")?.unwrap_or(false) {
        req.continue_final = true;
    }
    Ok(req)
}

fn named_choice(s: &str) -> Res<ToolChoice> {
    match s {
        "auto" => Ok(ToolChoice::Auto),
        "none" => Ok(ToolChoice::None),
        "required" => Ok(ToolChoice::Required),
        other => Err(format!(
            "tool_choice {other:?} is not supported (want auto, none or required)"
        )),
    }
}

fn json_schema(schema: Option<&Value>, at: &str) -> Res<ResponseFormat> {
    match schema {
        None | Some(Value::Null) => Ok(ResponseFormat::JsonObject),
        Some(s @ Value::Object(_)) => Ok(ResponseFormat::JsonSchema(s.clone())),
        Some(_) => Err(format!("{at} must be an object")),
    }
}

/// Chat content: a string, null, or a list of parts.
fn chat_content(v: Option<&Value>, at: &str, allow_media: bool) -> Res<Content> {
    let parts = match v {
        None | Some(Value::Null) => return Ok(Content::empty()),
        Some(Value::String(s)) => return Ok(Content::Text(s.clone())),
        Some(Value::Array(a)) => a,
        Some(_) => return Err(format!("{at}.content must be a string or a list of parts")),
    };
    let mut out = Vec::with_capacity(parts.len());
    for (j, p) in parts.iter().enumerate() {
        let here = format!("{at}.content.{j}");
        let po = as_obj(p, &here)?;
        if let Some(idx) = synthetic_media(po) {
            if !allow_media {
                return Err(format!(
                    "{here}: images and videos are only accepted in user and tool messages"
                ));
            }
            out.push(Part::Media(idx));
            continue;
        }
        match opt_str(po, "type")? {
            Some("text") | Some("input_text") | Some("output_text") => out.push(Part::Text(
                opt_str(po, "text")?
                    .ok_or(format!("{here}.text is required"))?
                    .to_string(),
            )),
            Some("refusal") => out.push(Part::Text(
                opt_str(po, "refusal")?.unwrap_or_default().to_string(),
            )),
            Some(other) => return Err(format!("{here}: content type {other:?} is not supported")),
            None => return Err(format!("{here}.type is required")),
        }
    }
    Ok(Content::Parts(out))
}

/// Sampling fields shared by the OpenAI protocols and Messages.
fn sampling(o: &Map<String, Value>, req: &mut ChatRequest) -> Res<()> {
    req.temperature = opt_f64(o, "temperature")?;
    req.top_p = opt_f64(o, "top_p")?;
    req.top_k = opt_i64(o, "top_k")?;
    req.min_p = opt_f64(o, "min_p")?;
    req.presence_penalty = opt_f64(o, "presence_penalty")?;
    req.frequency_penalty = opt_f64(o, "frequency_penalty")?;
    req.repetition_penalty = opt_f64(o, "repetition_penalty")?;
    req.seed = match o.get("seed") {
        None | Some(Value::Null) => None,
        Some(v) => Some(
            v.as_u64()
                .filter(|s| *s < 1 << 63)
                .ok_or("seed must be an integer in [0, 2^63)")?,
        ),
    };
    Ok(())
}

fn validate_sampling(req: &ChatRequest) -> Res<()> {
    let range = |name: &str, v: Option<f64>, lo: f64, hi: f64| match v {
        Some(x) if !(lo..=hi).contains(&x) => {
            Err(format!("{name} must be in [{lo}, {hi}], got {x}"))
        }
        _ => Ok(()),
    };
    range("temperature", req.temperature, 0.0, 2.0)?;
    range("min_p", req.min_p, 0.0, 1.0)?;
    range("presence_penalty", req.presence_penalty, -2.0, 2.0)?;
    range("frequency_penalty", req.frequency_penalty, -2.0, 2.0)?;
    if let Some(p) = req.top_p
        && !(p > 0.0 && p <= 1.0)
    {
        return Err(format!("top_p must be in (0, 1], got {p}"));
    }
    if let Some(k) = req.top_k
        && k < -1
    {
        return Err(format!("top_k must be -1, 0 or positive, got {k}"));
    }
    if let Some(r) = req.repetition_penalty
        && !(r > 0.0 && r <= 2.0)
    {
        return Err(format!("repetition_penalty must be in (0, 2], got {r}"));
    }
    Ok(())
}

/// Validate the tool choice against the tools; a named function keeps only
/// that tool.
fn resolve_tool_choice(req: &mut ChatRequest) -> Res<()> {
    if let ToolChoice::Function(name) = &req.tool_choice {
        let tool = req
            .tools
            .iter()
            .find(|t| tool_name(t) == name)
            .cloned()
            .ok_or_else(|| format!("tool_choice names {name:?}, which is not among the tools"))?;
        req.tools = vec![tool];
    }
    if matches!(
        req.tool_choice,
        ToolChoice::Required | ToolChoice::Function(_)
    ) && req.tools.is_empty()
    {
        return Err("tool_choice requires a tool call, but the request declares no tools".into());
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Anthropic Messages
// ---------------------------------------------------------------------------

fn messages(o: &Map<String, Value>) -> Res<ChatRequest> {
    let mut req = ChatRequest::new();
    match o.get("system") {
        None | Some(Value::Null) => {}
        Some(Value::String(s)) => req
            .messages
            .push(Message::new("system", Content::Text(s.clone()))),
        Some(Value::Array(blocks)) => {
            let mut parts = Vec::new();
            for (j, b) in blocks.iter().enumerate() {
                let b = as_obj(b, &format!("system.{j}"))?;
                parts.push(Part::Text(
                    opt_str(b, "text")?
                        .ok_or(format!("system.{j}.text is required"))?
                        .to_string(),
                ));
            }
            req.messages
                .push(Message::new("system", Content::Parts(parts)));
        }
        Some(_) => return Err("system must be a string or a list of text blocks".into()),
    }
    let msgs = opt_array(o, "messages")?.ok_or("messages is required")?;
    for (i, m) in msgs.iter().enumerate() {
        let at = format!("messages.{i}");
        let m = as_obj(m, &at)?;
        let role = opt_str(m, "role")?.ok_or(format!("{at}.role is required"))?;
        let blocks: Vec<Value> = match m.get("content") {
            Some(Value::String(s)) => vec![json!({"type": "text", "text": s})],
            Some(Value::Array(a)) => a.clone(),
            _ => return Err(format!("{at}.content must be a string or a list of blocks")),
        };
        match role {
            "user" => anthropic_user(&blocks, &at, &mut req.messages)?,
            "assistant" => {
                let msg = anthropic_assistant(&blocks, &at)?;
                req.messages.push(msg);
            }
            other => {
                return Err(format!(
                    "{at}.role {other:?} is not supported (want user or assistant)"
                ));
            }
        }
    }
    // A final assistant message is continued (Anthropic prefill).
    if matches!(req.messages.last(), Some(m) if m.role == "assistant") {
        req.continue_final = true;
    }

    for (i, t) in opt_array(o, "tools")?.into_iter().flatten().enumerate() {
        let at = format!("tools.{i}");
        let to = as_obj(t, &at)?;
        match opt_str(to, "type")? {
            None | Some("custom") => {}
            Some(other) => return Err(format!("{at}: server tool {other:?} is not supported")),
        }
        let name = opt_str(to, "name")?.ok_or(format!("{at}.name is required"))?;
        validate_tool_name(name, &at)?;
        req.tools.push(function_tool(
            name,
            to.get("description"),
            to.get("input_schema"),
        ));
    }
    req.tool_choice = match opt_obj(o, "tool_choice")? {
        None => ToolChoice::Auto,
        Some(c) => match opt_str(c, "type")? {
            Some("auto") | None => ToolChoice::Auto,
            Some("any") => ToolChoice::Required,
            Some("none") => ToolChoice::None,
            Some("tool") => ToolChoice::Function(
                opt_str(c, "name")?
                    .ok_or("tool_choice.name is required")?
                    .to_string(),
            ),
            Some(other) => return Err(format!("tool_choice.type {other:?} is not supported")),
        },
    };
    req.max_tokens = opt_u32(o, "max_tokens")?;
    sampling(o, &mut req)?;
    req.stop = stop_list(o.get("stop_sequences"), "stop_sequences")?;
    req.stream = opt_bool(o, "stream")?.unwrap_or(false);
    req.include_usage = true;
    req.thinking = thinking_type(o)?;
    Ok(req)
}

/// An Anthropic user turn: its tool results become tool messages, the rest
/// one user message.
fn anthropic_user(blocks: &[Value], at: &str, out: &mut Vec<Message>) -> Res<()> {
    let mut parts = Vec::new();
    for (j, b) in blocks.iter().enumerate() {
        let here = format!("{at}.content.{j}");
        let bo = as_obj(b, &here)?;
        if let Some(idx) = synthetic_media(bo) {
            parts.push(Part::Media(idx));
            continue;
        }
        match opt_str(bo, "type")? {
            Some("text") => parts.push(Part::Text(
                opt_str(bo, "text")?
                    .ok_or(format!("{here}.text is required"))?
                    .to_string(),
            )),
            Some("tool_result") => {
                let content = match bo.get("content") {
                    None | Some(Value::Null) => Content::empty(),
                    Some(Value::String(s)) => Content::Text(s.clone()),
                    Some(Value::Array(inner)) => {
                        let mut ps = Vec::new();
                        for (k, ib) in inner.iter().enumerate() {
                            let ihere = format!("{here}.content.{k}");
                            let io = as_obj(ib, &ihere)?;
                            if let Some(idx) = synthetic_media(io) {
                                ps.push(Part::Media(idx));
                            } else if opt_str(io, "type")? == Some("text") {
                                ps.push(Part::Text(
                                    opt_str(io, "text")?.unwrap_or_default().to_string(),
                                ));
                            } else {
                                return Err(format!(
                                    "{ihere}: only text, image and video blocks are supported in tool results"
                                ));
                            }
                        }
                        Content::Parts(ps)
                    }
                    Some(_) => {
                        return Err(format!(
                            "{here}.content must be a string or a list of blocks"
                        ));
                    }
                };
                let mut msg = Message::new("tool", content);
                msg.tool_call_id = opt_str(bo, "tool_use_id")?.map(str::to_string);
                out.push(msg);
            }
            Some(other) => {
                return Err(format!(
                    "{here}: block type {other:?} is not supported in user messages"
                ));
            }
            None => return Err(format!("{here}.type is required")),
        }
    }
    if !parts.is_empty() {
        out.push(Message::new("user", Content::Parts(parts)));
    }
    Ok(())
}

fn anthropic_assistant(blocks: &[Value], at: &str) -> Res<Message> {
    let mut parts = Vec::new();
    let mut msg = Message::new("assistant", Content::empty());
    let mut reasoning: Option<String> = None;
    for (j, b) in blocks.iter().enumerate() {
        let here = format!("{at}.content.{j}");
        let bo = as_obj(b, &here)?;
        match opt_str(bo, "type")? {
            Some("text") => parts.push(Part::Text(
                opt_str(bo, "text")?
                    .ok_or(format!("{here}.text is required"))?
                    .to_string(),
            )),
            Some("thinking") => {
                let t = opt_str(bo, "thinking")?.unwrap_or_default();
                reasoning.get_or_insert_with(String::new).push_str(t);
            }
            Some("redacted_thinking") => {}
            Some("tool_use") => {
                let name = opt_str(bo, "name")?.ok_or(format!("{here}.name is required"))?;
                msg.tool_calls.push(ToolCall {
                    id: opt_str(bo, "id")?.unwrap_or_default().to_string(),
                    name: name.to_string(),
                    arguments: bo.get("input").cloned().unwrap_or_else(|| json!({})),
                });
            }
            Some(other) => {
                return Err(format!(
                    "{here}: block type {other:?} is not supported in assistant messages"
                ));
            }
            None => return Err(format!("{here}.type is required")),
        }
    }
    msg.content = Content::Parts(parts);
    msg.reasoning = reasoning;
    Ok(msg)
}

// ---------------------------------------------------------------------------
// OpenAI Responses
// ---------------------------------------------------------------------------

fn responses(o: &Map<String, Value>) -> Res<ChatRequest> {
    let mut req = ChatRequest::new();
    if o.get("previous_response_id").is_some_and(|v| !v.is_null()) {
        return Err(
            "previous_response_id is not supported: send the whole conversation in input".into(),
        );
    }
    if let Some(s) = opt_str(o, "instructions")? {
        req.messages
            .push(Message::new("system", Content::Text(s.to_string())));
    }
    match o.get("input") {
        None | Some(Value::Null) => {}
        Some(Value::String(s)) => req
            .messages
            .push(Message::new("user", Content::Text(s.clone()))),
        Some(Value::Array(items)) => responses_items(items, &mut req.messages)?,
        Some(_) => return Err("input must be a string or a list of items".into()),
    }
    for (i, t) in opt_array(o, "tools")?.into_iter().flatten().enumerate() {
        let at = format!("tools.{i}");
        let to = as_obj(t, &at)?;
        // Hosted tools (web search, file search, ...) need the hosted
        // service; they are dropped.
        if opt_str(to, "type")? != Some("function") {
            continue;
        }
        let name = opt_str(to, "name")?.ok_or(format!("{at}.name is required"))?;
        validate_tool_name(name, &at)?;
        req.tools.push(function_tool(
            name,
            to.get("description"),
            to.get("parameters"),
        ));
    }
    req.tool_choice = match o.get("tool_choice") {
        None | Some(Value::Null) => ToolChoice::Auto,
        Some(Value::String(s)) => named_choice(s)?,
        Some(Value::Object(c)) => match c.get("type").and_then(Value::as_str) {
            Some("function") => ToolChoice::Function(
                c.get("name")
                    .and_then(Value::as_str)
                    .ok_or("tool_choice.name is required")?
                    .to_string(),
            ),
            _ => return Err("tool_choice: only function tool choices are supported".into()),
        },
        Some(_) => return Err("tool_choice must be a string or an object".into()),
    };
    if let Some(text) = opt_obj(o, "text")?
        && let Some(f) = opt_obj(text, "format")?
    {
        req.response_format = match opt_str(f, "type")? {
            Some("text") | None => ResponseFormat::Text,
            Some("json_object") => ResponseFormat::JsonObject,
            Some("json_schema") => json_schema(f.get("schema"), "text.format.schema")?,
            Some(other) => return Err(format!("text.format.type {other:?} is not supported")),
        };
    }
    req.max_tokens = opt_u32(o, "max_output_tokens")?;
    sampling(o, &mut req)?;
    if let Some(n) = opt_i64(o, "top_logprobs")? {
        if !(0..=20).contains(&n) {
            return Err(format!("top_logprobs must be in [0, 20], got {n}"));
        }
        req.logprobs = true;
        req.top_logprobs = Some(n as u8);
    }
    req.stream = opt_bool(o, "stream")?.unwrap_or(false);
    req.include_usage = true;
    if let Some(r) = opt_obj(o, "reasoning")? {
        req.reasoning_effort = opt_str(r, "effort")?.map(str::to_string);
        req.thinking = effort_thinking(req.reasoning_effort.as_deref());
    }
    Ok(req)
}

fn responses_items(items: &[Value], out: &mut Vec<Message>) -> Res<()> {
    let mut pending_reasoning: Option<String> = None;
    // Whether the last message is an assistant turn that later tool calls
    // of the same turn join.
    let mut open_turn = false;
    for (i, item) in items.iter().enumerate() {
        let at = format!("input.{i}");
        let io = as_obj(item, &at)?;
        let typ = opt_str(io, "type")?.unwrap_or(if io.contains_key("role") {
            "message"
        } else {
            ""
        });
        match typ {
            "message" => {
                let role = match opt_str(io, "role")?.ok_or(format!("{at}.role is required"))? {
                    "user" => "user",
                    "assistant" => "assistant",
                    "system" | "developer" => "system",
                    other => return Err(format!("{at}.role {other:?} is not supported")),
                };
                let content = responses_content(io.get("content"), &at, role != "assistant")?;
                let mut msg = Message::new(role, content);
                if role == "assistant" {
                    msg.reasoning = pending_reasoning.take();
                }
                open_turn = role == "assistant";
                out.push(msg);
            }
            "function_call" => {
                let name = opt_str(io, "name")?.ok_or(format!("{at}.name is required"))?;
                let call = ToolCall {
                    id: opt_str(io, "call_id")?.unwrap_or_default().to_string(),
                    name: name.to_string(),
                    arguments: parse_arguments(opt_str(io, "arguments")?.unwrap_or("{}")),
                };
                if !open_turn {
                    let mut msg = Message::new("assistant", Content::empty());
                    msg.reasoning = pending_reasoning.take();
                    out.push(msg);
                    open_turn = true;
                }
                if let Some(last) = out.last_mut() {
                    last.tool_calls.push(call);
                }
            }
            "function_call_output" => {
                let content = match io.get("output") {
                    None | Some(Value::Null) => Content::empty(),
                    Some(Value::String(s)) => Content::Text(s.clone()),
                    Some(v @ Value::Array(_)) => {
                        responses_content(Some(v), &format!("{at}.output"), true)?
                    }
                    Some(_) => {
                        return Err(format!("{at}.output must be a string or a list of parts"));
                    }
                };
                let mut msg = Message::new("tool", content);
                msg.tool_call_id = opt_str(io, "call_id")?.map(str::to_string);
                out.push(msg);
                open_turn = false;
            }
            "reasoning" => {
                let mut text = String::new();
                for key in ["content", "summary"] {
                    for (j, c) in opt_array(io, key)?.into_iter().flatten().enumerate() {
                        if let Some(t) = c.get("text").and_then(Value::as_str) {
                            if !text.is_empty() || j > 0 {
                                text.push('\n');
                            }
                            text.push_str(t);
                        }
                    }
                    if !text.is_empty() {
                        break;
                    }
                }
                pending_reasoning = Some(text);
            }
            // Item references and hosted-tool items need server state.
            _ => {}
        }
    }
    Ok(())
}

fn responses_content(v: Option<&Value>, at: &str, allow_media: bool) -> Res<Content> {
    match v {
        None | Some(Value::Null) => Ok(Content::empty()),
        Some(Value::String(s)) => Ok(Content::Text(s.clone())),
        Some(Value::Array(_)) => chat_content(v, at, allow_media),
        Some(_) => Err(format!("{at}.content must be a string or a list of parts")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(s: &str) -> Content {
        Content::Text(s.into())
    }

    #[test]
    fn chat_messages_tools_and_options() {
        let r = convert(
            PROTOCOL_CHAT_COMPLETIONS,
            &json!({"model": "m", "messages": [
                {"role": "developer", "content": "Be brief."},
                {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
                {"role": "assistant", "content": null, "reasoning_content": "think",
                 "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{\"x\":1}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "42"}],
                "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
                "max_tokens": 9, "max_completion_tokens": 7, "temperature": 0.5, "top_k": 20, "seed": 3,
                "stop": "END", "logprobs": true, "top_logprobs": 2, "stream": true,
                "stream_options": {"include_usage": true}, "chat_template_kwargs": {"enable_thinking": false}}),
        )
        .unwrap();
        assert_eq!(r.messages.len(), 4);
        assert_eq!(r.messages[0].role, "system");
        assert_eq!(r.messages[1].content.joined_text().unwrap(), "a\nb");
        let a = &r.messages[2];
        assert_eq!(a.reasoning.as_deref(), Some("think"));
        assert_eq!(a.tool_calls[0].arguments, json!({"x": 1}));
        assert_eq!(r.messages[3].tool_call_id.as_deref(), Some("c1"));
        assert_eq!(r.max_tokens, Some(7));
        assert_eq!(
            (r.temperature, r.top_k, r.seed),
            (Some(0.5), Some(20), Some(3))
        );
        assert_eq!(r.stop, vec!["END".to_string()]);
        assert_eq!((r.logprobs, r.top_logprobs), (true, Some(2)));
        assert!(r.stream && r.include_usage);
        assert_eq!(r.thinking, Some(false));
        assert_eq!(r.tool_choice, ToolChoice::Auto);
    }

    #[test]
    fn named_tool_choice_keeps_one_tool() {
        let tools = json!([{"type": "function", "function": {"name": "a"}}, {"type": "function", "function": {"name": "b"}}]);
        let r = convert(
            PROTOCOL_CHAT_COMPLETIONS,
            &json!({"messages": [{"role": "user", "content": "x"}], "tools": tools,
                    "tool_choice": {"type": "function", "function": {"name": "b"}}}),
        )
        .unwrap();
        assert_eq!(r.tools.len(), 1);
        assert_eq!(tool_name(&r.tools[0]), "b");
        assert_eq!(r.tool_choice.name(), "required");
        let err = convert(
            PROTOCOL_CHAT_COMPLETIONS,
            &json!({"messages": [{"role": "user", "content": "x"}], "tools": tools,
                    "tool_choice": {"type": "function", "function": {"name": "zz"}}}),
        )
        .unwrap_err();
        assert!(err.contains("not among the tools"), "{err}");
    }

    #[test]
    fn invalid_requests_are_rejected() {
        for (body, want) in [
            (json!({"messages": []}), "no messages"),
            (
                json!({"messages": [{"role": "robot", "content": "x"}]}),
                "not supported",
            ),
            (
                json!({"messages": [{"role": "user", "content": "x"}], "n": 2}),
                "n must be 1",
            ),
            (
                json!({"messages": [{"role": "user", "content": "x"}], "temperature": 3}),
                "temperature",
            ),
            (
                json!({"messages": [{"role": "user", "content": "x"}], "top_logprobs": 3}),
                "requires logprobs",
            ),
            (
                json!({"messages": [{"role": "user", "content": "x"}], "tool_choice": "required"}),
                "declares no tools",
            ),
            (
                json!({"messages": [{"role": "user", "content": "x"}], "max_tokens": 0}),
                "at least 1",
            ),
            (
                json!({"messages": [{"role": "user", "content": [{"type": "input_audio"}]}]}),
                "not supported",
            ),
            (
                json!({"messages": [{"role": "user", "content": "x"}], "continue_final_message": true}),
                "final assistant",
            ),
        ] {
            let err = convert(PROTOCOL_CHAT_COMPLETIONS, &body).unwrap_err();
            assert!(err.contains(want), "{body}: {err}");
        }
    }

    #[test]
    fn anthropic_messages_map_to_chat_messages() {
        let r = convert(
            PROTOCOL_MESSAGES,
            &json!({"model": "m", "max_tokens": 64, "system": [{"type": "text", "text": "S"}],
                    "messages": [
                        {"role": "user", "content": "q"},
                        {"role": "assistant", "content": [
                            {"type": "thinking", "thinking": "hmm", "signature": "x"},
                            {"type": "text", "text": "Let me look."},
                            {"type": "tool_use", "id": "t1", "name": "f", "input": {"x": 1}}]},
                        {"role": "user", "content": [
                            {"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "42"}]},
                            {"type": "text", "text": "thanks"}]}],
                    "tools": [{"name": "f", "description": "d", "input_schema": {"type": "object"}}],
                    "tool_choice": {"type": "any"}, "stop_sequences": ["X"], "top_k": 5,
                    "thinking": {"type": "disabled"}}),
        )
        .unwrap();
        let roles: Vec<&str> = r.messages.iter().map(|m| m.role).collect();
        assert_eq!(roles, vec!["system", "user", "assistant", "tool", "user"]);
        assert_eq!(r.messages[2].reasoning.as_deref(), Some("hmm"));
        assert_eq!(r.messages[2].tool_calls[0].arguments, json!({"x": 1}));
        assert_eq!(r.messages[3].tool_call_id.as_deref(), Some("t1"));
        assert_eq!(
            r.tools[0],
            json!({"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object"}}})
        );
        assert_eq!(r.tool_choice, ToolChoice::Required);
        assert_eq!((r.max_tokens, r.top_k), (Some(64), Some(5)));
        assert_eq!(r.thinking, Some(false));
        assert!(!r.continue_final);
        // A final assistant message is continued.
        let prefill = convert(
            PROTOCOL_MESSAGES,
            &json!({"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "The"}]}),
        )
        .unwrap();
        assert!(prefill.continue_final);
    }

    #[test]
    fn responses_items_map_to_chat_messages() {
        let r = convert(
            PROTOCOL_RESPONSES,
            &json!({"model": "m", "instructions": "S", "input": [
                {"role": "user", "content": [{"type": "input_text", "text": "q"}]},
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "plan"}]},
                {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{\"x\":1}"},
                {"type": "function_call", "call_id": "c2", "name": "f", "arguments": "{\"x\":2}"},
                {"type": "function_call_output", "call_id": "c1", "output": "1"},
                {"type": "function_call_output", "call_id": "c2", "output": "2"},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]}],
                "tools": [{"type": "function", "name": "f", "parameters": {"type": "object"}}, {"type": "web_search"}],
                "max_output_tokens": 5, "reasoning": {"effort": "none"}, "top_logprobs": 1,
                "text": {"format": {"type": "json_schema", "name": "x", "schema": {"type": "object"}}}}),
        )
        .unwrap();
        let roles: Vec<&str> = r.messages.iter().map(|m| m.role).collect();
        assert_eq!(
            roles,
            vec!["system", "user", "assistant", "tool", "tool", "assistant"]
        );
        assert_eq!(r.messages[2].reasoning.as_deref(), Some("plan"));
        assert_eq!(r.messages[2].tool_calls.len(), 2);
        assert_eq!(r.messages[2].content, Content::empty());
        assert_eq!(r.messages[5].content.joined_text().unwrap(), "done");
        assert_eq!(r.tools.len(), 1);
        assert_eq!(r.max_tokens, Some(5));
        assert_eq!(r.thinking, Some(false));
        assert!(r.logprobs);
        assert_eq!(
            r.response_format,
            ResponseFormat::JsonSchema(json!({"type": "object"}))
        );
        let plain = convert(PROTOCOL_RESPONSES, &json!({"input": "hi"})).unwrap();
        assert_eq!(plain.messages[0].content, text("hi"));
        assert!(
            convert(
                PROTOCOL_RESPONSES,
                &json!({"input": "hi", "previous_response_id": "r"})
            )
            .unwrap_err()
            .contains("previous_response_id")
        );
    }
}

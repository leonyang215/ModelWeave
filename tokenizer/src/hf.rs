//! Renderer of chat-template models (Qwen3, Qwen3-VL, any Hugging Face chat
//! template): API request -> neutral chat request (`chat`) -> the model's
//! Jinja chat template (`template`) -> token IDs, with images and videos
//! planned and expanded like the model's processor does.
//!
//! Control tokens only come from the template. Every string of the request
//! is escaped before rendering, so a special-token spelling a client sends
//! (`<|im_start|>`, `<|image_pad|>`, ...) stays text: it is tokenized like
//! any other text and can neither forge a turn nor a media placeholder.
//! Tokens the tokenizer marks as non-special (`<think>`, `<tool_call>`) are
//! parsed wherever they appear, as Hugging Face tokenizers do.

use std::borrow::Cow;

use aho_corasick::{AhoCorasick, MatchKind};
use serde_json::{Map, Value, json};
use tokenizers::Tokenizer;

use crate::chat::{self, ChatRequest, Content, Part, ResponseFormat, ToolChoice};
use crate::error::ApiError;
use crate::media::{self, MediaInput, MediaLayout, Modality, Slot};
use crate::metrics;
use crate::model::{ModelMeta, Stopwatch, TemplateSource, segments_by_marker};
use crate::protocol::*;
use crate::template::{ChatTemplate, ContentFormat};
use crate::vision::VisionSpec;

/// How a model's thinking is controlled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Thinking {
    /// The model does not think.
    None,
    /// A chat template variable switches thinking per request.
    Toggle,
    /// The model always thinks.
    Always,
}

impl Thinking {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "none" => Ok(Self::None),
            "toggle" => Ok(Self::Toggle),
            "always" => Ok(Self::Always),
            other => Err(format!(
                "unknown thinking mode {other:?} (want toggle, always or none)"
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Toggle => "toggle",
            Self::Always => "always",
        }
    }
}

/// Settings of a chat-template model.
#[derive(Debug, Clone)]
pub struct TemplateSettings {
    pub thinking: Thinking,
    pub default_thinking: bool,
    pub thinking_kwarg: String,
    pub reasoning_parser: Option<String>,
    pub tool_call_parser: Option<String>,
    /// Stop tokens; empty: generation_config.json, else the EOS token.
    pub stop_tokens: Vec<String>,
    /// `None`: detected from the template.
    pub content_format: Option<ContentFormat>,
    pub default_max_tokens: u32,
    pub thinking_max_tokens: u32,
    /// Token that opens every message (ChatML `<|im_start|>`), for segments.
    pub role_marker: Option<String>,
    /// Reasoning delimiters (`<think>`, `</think>`).
    pub think_tokens: Option<(String, String)>,
    pub vision: Option<VisionSpec>,
}

/// Special (control) tokens of a tokenizer: they only come from templates.
struct Specials {
    matcher: Option<AhoCorasick>,
    ids: Vec<u32>,
}

impl Specials {
    fn new(tokenizer: &Tokenizer) -> Result<Self, String> {
        let mut tokens: Vec<(String, u32)> = tokenizer
            .get_added_tokens_decoder()
            .into_iter()
            .filter(|(_, t)| t.special && !t.content.is_empty())
            .map(|(id, t)| (t.content, id))
            .collect();
        tokens.sort();
        let matcher = (!tokens.is_empty())
            .then(|| {
                AhoCorasick::builder()
                    .match_kind(MatchKind::LeftmostLongest)
                    .build(tokens.iter().map(|(s, _)| s.as_str()))
                    .map_err(|e| e.to_string())
            })
            .transpose()?;
        Ok(Self {
            matcher,
            ids: tokens.into_iter().map(|(_, id)| id).collect(),
        })
    }

    /// `s` with a marker after the first character of every special-token
    /// spelling, so the template output does not hold it any more.
    fn escape(&self, s: &str, marker: char) -> Option<String> {
        let m = self.matcher.as_ref()?;
        let mut found = m.find_iter(s).peekable();
        found.peek()?;
        let mut out = String::with_capacity(s.len() + 8);
        let mut last = 0;
        for hit in found {
            let first = s[hit.start()..].chars().next().map_or(1, char::len_utf8);
            out.push_str(&s[last..hit.start() + first]);
            out.push(marker);
            last = hit.start() + first;
        }
        out.push_str(&s[last..]);
        Some(out)
    }

    fn escape_value(&self, v: &mut Value, marker: char) {
        match v {
            Value::String(s) => {
                if let Some(e) = self.escape(s, marker) {
                    *s = e;
                }
            }
            Value::Array(a) => a.iter_mut().for_each(|x| self.escape_value(x, marker)),
            Value::Object(o) => o.values_mut().for_each(|x| self.escape_value(x, marker)),
            _ => {}
        }
    }
}

/// Token IDs of a vision model's placeholders.
struct VisionTokens {
    spec: VisionSpec,
    image_pad: u32,
    video_pad: u32,
    start: u32,
    end: u32,
}

/// Prompt renderer of one chat-template model.
pub struct TemplateRenderer {
    pub settings: TemplateSettings,
    template: ChatTemplate,
    content_format: ContentFormat,
    bos_token: Option<String>,
    eos_token: Option<String>,
    specials: Specials,
    /// The tokenizer with special tokens disabled: plain text.
    plain: Tokenizer,
    vision: Option<VisionTokens>,
    role_marker: Option<u32>,
    roles: Vec<(u32, &'static str)>,
    pub stop_ids: Vec<u32>,
    pub think_ids: (Option<u32>, Option<u32>),
    sampling: Sampling,
}

/// What a chat template renders.
struct Variables {
    /// The template variables, request strings escaped.
    vars: Value,
    /// The chat template kwargs as rendered (unescaped).
    kwargs: Map<String, Value>,
    /// Media indices in the order the messages hold them.
    media_order: Vec<usize>,
}

/// Sampling defaults of generation_config.json.
#[derive(Debug, Clone, Default)]
struct Sampling {
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<i64>,
    min_p: Option<f32>,
    repetition_penalty: Option<f32>,
}

impl TemplateRenderer {
    pub fn new(
        tokenizer: &Tokenizer,
        source: TemplateSource,
        settings: TemplateSettings,
        generation: Option<&Value>,
    ) -> Result<Self, String> {
        let template = ChatTemplate::new(source.template)?;
        let content_format = settings.content_format.unwrap_or(template.content_format);
        let id = |s: &str| {
            tokenizer
                .token_to_id(s)
                .ok_or_else(|| format!("the tokenizer has no token {s:?}"))
        };
        let mut stop_ids = Vec::new();
        for t in &settings.stop_tokens {
            stop_ids.push(id(t)?);
        }
        if stop_ids.is_empty()
            && let Some(g) = generation
        {
            match g.get("eos_token_id") {
                Some(Value::Number(n)) => stop_ids.extend(n.as_u64().map(|v| v as u32)),
                Some(Value::Array(a)) => {
                    stop_ids.extend(a.iter().filter_map(Value::as_u64).map(|v| v as u32))
                }
                _ => {}
            }
        }
        if stop_ids.is_empty()
            && let Some(e) = &source.eos_token
        {
            stop_ids.push(id(e)?);
        }
        if stop_ids.is_empty() {
            return Err("no stop tokens: set stop_tokens or provide generation_config.json".into());
        }
        let vision = match &settings.vision {
            None => None,
            Some(spec) => Some(VisionTokens {
                spec: spec.clone(),
                image_pad: id("<|image_pad|>")?,
                video_pad: id("<|video_pad|>")?,
                start: id("<|vision_start|>")?,
                end: id("<|vision_end|>")?,
            }),
        };
        let role_marker = settings.role_marker.as_deref().map(id).transpose()?;
        let mut plain = tokenizer.clone();
        plain.set_encode_special_tokens(true);
        let mut roles = Vec::new();
        for role in ["system", "user", "assistant", "tool"] {
            if let Ok(e) = plain.encode_fast(role, false)
                && let [t] = e.get_ids()
            {
                roles.push((*t, role));
            }
        }
        let think_ids = match &settings.think_tokens {
            Some((a, b)) => (tokenizer.token_to_id(a), tokenizer.token_to_id(b)),
            None => (None, None),
        };
        let f32_of = |k: &str| {
            generation
                .and_then(|g| g.get(k))
                .and_then(Value::as_f64)
                .map(|v| v as f32)
        };
        let sampling = Sampling {
            temperature: f32_of("temperature"),
            top_p: f32_of("top_p"),
            top_k: generation
                .and_then(|g| g.get("top_k"))
                .and_then(Value::as_i64),
            min_p: f32_of("min_p"),
            repetition_penalty: f32_of("repetition_penalty").filter(|r| *r != 1.0),
        };
        Ok(Self {
            specials: Specials::new(tokenizer)?,
            settings,
            template,
            content_format,
            bos_token: source.bos_token,
            eos_token: source.eos_token,
            plain,
            vision,
            role_marker,
            roles,
            stop_ids,
            think_ids,
            sampling,
        })
    }

    /// Encode plain text: special-token spellings stay text.
    pub fn encode_plain(&self, text: &str) -> Result<Vec<u32>, String> {
        self.plain
            .encode_fast(text, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| e.to_string())
    }

    pub fn render(&self, meta: &ModelMeta, req: &RenderRequest) -> Result<RenderResult, ApiError> {
        let protocol = req.protocol.as_str();
        let bad = |stage: &str, msg: String| {
            metrics::error(&meta.name, stage);
            ApiError::bad_request(protocol, msg)
        };
        let mut sw = Stopwatch::new();
        let mut body: Value = serde_json::from_str(req.body.get())
            .map_err(|e| bad("parse", format!("invalid JSON request: {e}")))?;
        let inputs = media::extract(protocol, &mut body).map_err(|e| bad("media", e))?;
        if !inputs.is_empty() && self.vision.is_none() {
            return Err(bad(
                "media",
                format!("model {} does not accept images or videos", meta.name),
            ));
        }
        sw.lap("parse");

        let chat = chat::convert(protocol, &body).map_err(|e| bad("convert", e))?;
        let thinking = match self.settings.thinking {
            Thinking::None => false,
            Thinking::Always => true,
            Thinking::Toggle => chat.thinking.unwrap_or(self.settings.default_thinking),
        };
        let marker = free_marker(&chat);
        let Variables {
            vars,
            kwargs,
            media_order,
        } = self
            .variables(&chat, &inputs, thinking, marker)
            .map_err(|e| bad("convert", e))?;
        sw.lap("convert");

        let mut prompt = self
            .template
            .render(&vars)
            .map_err(|e| bad("render", format!("chat template: {}", e.0)))?;
        if chat.continue_final {
            prompt = self
                .continue_final(prompt, &chat, marker)
                .map_err(|e| bad("render", e))?;
        }
        sw.lap("render");

        let raw_ids = self.encode_prompt(&prompt, marker).map_err(|e| {
            metrics::error(&meta.name, "tokenize");
            ApiError::internal(protocol, format!("tokenize: {e}"))
        })?;
        sw.lap("tokenize");

        let (token_ids, items) = match &self.vision {
            Some(v) if !inputs.is_empty() => {
                let planned = media::plan_all(&v.spec, &inputs).map_err(|e| bad("media", e))?;
                let slots = self
                    .slots(v, &raw_ids, &inputs, &media_order)
                    .map_err(|e| bad("media", e))?;
                let layout = MediaLayout {
                    #[cfg(test)]
                    image_placeholder: vec![v.image_pad],
                    #[cfg(test)]
                    video_placeholder: vec![v.start, v.video_pad, v.end],
                    frame_text: |t| format!("<{t:.1} seconds>"),
                    frame_open: Some(v.start),
                    frame_close: Some(v.end),
                };
                let encode = |t: &str| self.encode_plain(t);
                let out =
                    media::expand(&raw_ids, &slots, &planned, &layout, &encode).map_err(|e| {
                        metrics::error(&meta.name, "media");
                        ApiError::internal(protocol, e)
                    })?;
                for p in &planned {
                    metrics::media(&meta.name, p.input.modality.name());
                }
                out
            }
            _ => (raw_ids, Vec::new()),
        };
        sw.lap("media");

        let prompt_tokens = token_ids.len();
        let context_limit = req.context_length.unwrap_or(meta.context_length);
        let max_tokens = resolve_max_tokens(
            chat.max_tokens,
            prompt_tokens as u32,
            context_limit,
            if thinking {
                self.settings.thinking_max_tokens
            } else {
                self.settings.default_max_tokens
            },
        )
        .map_err(|e| bad("convert", e))?;

        let use_tools = chat.tool_choice != ToolChoice::None && !chat.tools.is_empty();
        let tools = if use_tools {
            chat.tools.clone()
        } else {
            Vec::new()
        };
        let reasoning = thinking && self.settings.reasoning_parser.is_some();
        let stage = reasoning.then(|| {
            if chat.continue_final {
                if self.open_reasoning(&chat) {
                    "reasoning_continuation"
                } else {
                    "content_continuation"
                }
            } else {
                "reasoning_start"
            }
        });
        let output_constraint = if use_tools && self.settings.tool_call_parser.is_some() {
            Some(OutputConstraint {
                kind: if chat.tool_choice == ToolChoice::Auto {
                    "tool_call"
                } else {
                    "tool_call_required"
                },
                schema: Value::Array(tools.clone()).to_string(),
            })
        } else {
            match &chat.response_format {
                ResponseFormat::Text => None,
                ResponseFormat::JsonObject => Some(OutputConstraint {
                    kind: "json",
                    schema: String::new(),
                }),
                ResponseFormat::JsonSchema(s) => Some(OutputConstraint {
                    kind: "json",
                    schema: s.to_string(),
                }),
                ResponseFormat::Regex(r) => Some(OutputConstraint {
                    kind: "regex",
                    schema: r.clone(),
                }),
            }
        };
        let segments = match self.role_marker {
            Some(m) => segments_by_marker(&token_ids, m, &self.roles),
            None => Vec::new(),
        };
        let n = token_ids.len();
        let primer = match segments.last() {
            Some(seg) if seg.role == "assistant" && !chat.continue_final => n - seg.offset,
            _ => 0,
        };
        let multimodal = (!items.is_empty()).then_some(Multimodal {
            pad_shift: media::PAD_SHIFT,
            items,
        });
        let s = &self.sampling;
        let f32_opt = |v: Option<f64>| v.map(|x| x as f32);
        let timing = sw.finish(&meta.name, protocol, prompt_tokens);
        Ok(RenderResult {
            request_id: req.request_id.clone(),
            protocol: protocol.to_string(),
            model: meta.name.clone(),
            model_version: meta.family.name().into(),
            prompt_tokens,
            token_ids,
            multimodal,
            conversation: ConversationInfo {
                thinking_mode: reasoning,
                reasoning_effort: if reasoning {
                    chat.reasoning_effort.clone()
                } else {
                    None
                },
                tools,
                tool_choice: chat.tool_choice.name(),
                response_format: chat.response_format.name(),
                chat_template_kwargs: kwargs,
            },
            inference_options: InferenceOptions {
                max_tokens,
                temperature: f32_opt(chat.temperature).or(s.temperature),
                top_p: f32_opt(chat.top_p).or(s.top_p),
                top_k: chat.top_k.or(s.top_k),
                min_p: f32_opt(chat.min_p).or(s.min_p),
                presence_penalty: f32_opt(chat.presence_penalty),
                frequency_penalty: f32_opt(chat.frequency_penalty),
                repetition_penalty: f32_opt(chat.repetition_penalty).or(s.repetition_penalty),
                seed: chat.seed,
                logprobs: chat.logprobs,
                top_logprobs: chat.top_logprobs,
                stop_token_ids: self.stop_ids.clone(),
                output_constraint,
            },
            parsing_options: ParsingOptions {
                reasoning_parser: self.settings.reasoning_parser.clone(),
                tool_call_parser: self.settings.tool_call_parser.clone(),
                parse_tool_calls: use_tools && self.settings.tool_call_parser.is_some(),
                starts_in_tool_calls: false,
                parse_json_output: chat.response_format == ResponseFormat::JsonObject,
                thinking_initial_stage: stage,
                stop_sequences: chat.stop.clone(),
            },
            stream: chat.stream,
            response_options: ResponseOptions {
                include_usage: chat.include_usage,
            },
            render_artifact: RenderArtifact {
                version: 2,
                encoding: meta.family.name().into(),
                token_layout: TokenLayout {
                    cacheable_prefix: Span {
                        offset: 0,
                        length: n - primer,
                    },
                    generation_primer: Span {
                        offset: n - primer,
                        length: primer,
                    },
                },
                segments,
                prefix_continuation: chat.continue_final,
            },
            timing_us: timing,
        })
    }

    /// The template variables of a request.
    fn variables(
        &self,
        chat: &ChatRequest,
        inputs: &[MediaInput],
        thinking: bool,
        marker: char,
    ) -> Result<Variables, String> {
        let mut order = Vec::new();
        let mut messages = Vec::with_capacity(chat.messages.len());
        for (i, m) in chat.messages.iter().enumerate() {
            let mut o = Map::new();
            o.insert("role".into(), json!(m.role));
            let content = self
                .content(&m.content, inputs, &mut order)
                .map_err(|e| format!("messages.{i}: {e}"))?;
            o.insert("content".into(), content);
            if let Some(r) = &m.reasoning {
                o.insert("reasoning_content".into(), json!(r));
            }
            if !m.tool_calls.is_empty() {
                let calls: Vec<Value> = m
                    .tool_calls
                    .iter()
                    .map(|c| {
                        json!({"id": c.id, "type": "function",
                               "function": {"name": c.name, "arguments": c.arguments}})
                    })
                    .collect();
                o.insert("tool_calls".into(), Value::Array(calls));
            }
            if let Some(id) = &m.tool_call_id {
                o.insert("tool_call_id".into(), json!(id));
            }
            if let Some(n) = &m.name {
                o.insert("name".into(), json!(n));
            }
            messages.push(Value::Object(o));
        }
        let mut kwargs = chat.template_kwargs.clone();
        if self.settings.thinking == Thinking::Toggle {
            kwargs.insert(self.settings.thinking_kwarg.clone(), json!(thinking));
        }
        let mut vars = kwargs.clone();
        for k in [
            "messages",
            "tools",
            "add_generation_prompt",
            "bos_token",
            "eos_token",
        ] {
            vars.remove(k);
        }
        vars.insert("messages".into(), Value::Array(messages));
        if chat.tool_choice != ToolChoice::None && !chat.tools.is_empty() {
            vars.insert("tools".into(), Value::Array(chat.tools.clone()));
        }
        let mut vars = Value::Object(vars);
        self.specials.escape_value(&mut vars, marker);
        let o = vars.as_object_mut().expect("object");
        o.insert("add_generation_prompt".into(), json!(!chat.continue_final));
        if let Some(b) = &self.bos_token {
            o.insert("bos_token".into(), json!(b));
        }
        if let Some(e) = &self.eos_token {
            o.insert("eos_token".into(), json!(e));
        }
        Ok(Variables {
            vars,
            kwargs,
            media_order: order,
        })
    }

    /// Message content in the template's format.
    fn content(
        &self,
        c: &Content,
        inputs: &[MediaInput],
        order: &mut Vec<usize>,
    ) -> Result<Value, String> {
        match (c, self.content_format) {
            (Content::Text(s), _) => Ok(json!(s)),
            (Content::Parts(parts), ContentFormat::String) => {
                c.joined_text().map(Value::String).ok_or_else(|| {
                    format!("the chat template takes text only ({} parts)", parts.len())
                })
            }
            (Content::Parts(parts), ContentFormat::Parts) => Ok(Value::Array(
                parts
                    .iter()
                    .map(|p| match p {
                        Part::Text(t) => json!({"type": "text", "text": t}),
                        Part::Media(i) => {
                            order.push(*i);
                            let kind = inputs.get(*i).map_or("image", |m| m.modality.name());
                            json!({"type": kind})
                        }
                    })
                    .collect(),
            )),
        }
    }

    /// Cut the prompt after the final message's text, so the model continues
    /// it (transformers' `continue_final_message`).
    fn continue_final(
        &self,
        prompt: String,
        chat: &ChatRequest,
        marker: char,
    ) -> Result<String, String> {
        let last = chat.messages.last().ok_or("no final message")?;
        let text = match &last.content {
            Content::Text(s) => s.clone(),
            Content::Parts(parts) => parts
                .iter()
                .rev()
                .find_map(|p| match p {
                    Part::Text(t) => Some(t.clone()),
                    Part::Media(_) => None,
                })
                .unwrap_or_default(),
        };
        let text = self.specials.escape(&text, marker).unwrap_or(text);
        let stripped = text.trim();
        if stripped.is_empty() {
            return Err("the final assistant message has no text to continue".into());
        }
        let loc = prompt.rfind(stripped).ok_or(
            "the chat template does not render the final message, so it cannot be continued",
        )?;
        let lstripped = text.trim_start();
        let end = if prompt[loc..].starts_with(lstripped) {
            loc + lstripped.len()
        } else {
            loc + stripped.len()
        };
        let mut prompt = prompt;
        prompt.truncate(end);
        Ok(prompt)
    }

    /// Whether the continued final message is inside an unclosed reasoning
    /// block.
    fn open_reasoning(&self, chat: &ChatRequest) -> bool {
        let Some((start, end)) = &self.settings.think_tokens else {
            return false;
        };
        let text = chat
            .messages
            .last()
            .and_then(|m| m.content.joined_text())
            .unwrap_or_default();
        match (text.rfind(start.as_str()), text.rfind(end.as_str())) {
            (Some(s), Some(e)) => s > e,
            (Some(_), None) => true,
            _ => false,
        }
    }

    /// Tokenize the rendered prompt: special-token spellings the template
    /// wrote become control tokens, the text between them (marker removed)
    /// is encoded with special tokens disabled.
    fn encode_prompt(&self, prompt: &str, marker: char) -> Result<Vec<u32>, String> {
        let mut regions: Vec<Cow<str>> = Vec::new();
        let mut specials: Vec<u32> = Vec::new();
        let mut last = 0;
        if let Some(m) = &self.specials.matcher {
            for hit in m.find_iter(prompt) {
                regions.push(strip_marker(&prompt[last..hit.start()], marker));
                specials.push(self.specials.ids[hit.pattern().as_usize()]);
                last = hit.end();
            }
        }
        regions.push(strip_marker(&prompt[last..], marker));
        let inputs: Vec<&str> = regions.iter().map(|r| r.as_ref()).collect();
        let encodings = self
            .plain
            .encode_batch_fast(inputs, false)
            .map_err(|e| e.to_string())?;
        let mut ids = Vec::with_capacity(prompt.len() / 3);
        for (i, enc) in encodings.iter().enumerate() {
            ids.extend_from_slice(enc.get_ids());
            if let Some(s) = specials.get(i) {
                ids.push(*s);
            }
        }
        Ok(ids)
    }

    /// The rendered media placeholders: `<|image_pad|>` for an image,
    /// `<|vision_start|><|video_pad|><|vision_end|>` for a video, matched in
    /// order with the media of the messages.
    fn slots(
        &self,
        v: &VisionTokens,
        ids: &[u32],
        inputs: &[MediaInput],
        order: &[usize],
    ) -> Result<Vec<Slot>, String> {
        let mut slots = Vec::with_capacity(order.len());
        let mut i = 0;
        while i < ids.len() {
            let t = ids[i];
            let (start, len, modality) = if t == v.image_pad {
                (i, 1, Modality::Image)
            } else if t == v.video_pad {
                if i == 0 || ids[i - 1] != v.start || ids.get(i + 1) != Some(&v.end) {
                    return Err(
                        "the chat template renders a video placeholder outside a vision block"
                            .into(),
                    );
                }
                (i - 1, 3, Modality::Video)
            } else {
                i += 1;
                continue;
            };
            let k = slots.len();
            let media = *order.get(k).ok_or_else(|| {
                format!("the chat template renders more media placeholders than the request has media ({})", order.len())
            })?;
            if inputs[media].modality != modality {
                return Err(format!(
                    "{}: the chat template renders a {} placeholder for this {}",
                    inputs[media].path,
                    modality.name(),
                    inputs[media].modality.name()
                ));
            }
            slots.push(Slot { start, len, media });
            i = start + len;
        }
        if slots.len() != order.len() || order.len() != inputs.len() {
            return Err(format!(
                "the chat template renders {} of the request's {} images and videos (media are accepted in user and tool messages)",
                slots.len(),
                inputs.len()
            ));
        }
        Ok(slots)
    }
}

fn strip_marker(s: &str, marker: char) -> Cow<'_, str> {
    if s.contains(marker) {
        Cow::Owned(s.replace(marker, ""))
    } else {
        Cow::Borrowed(s)
    }
}

/// A private-use character that no string of the request contains.
fn free_marker(chat: &ChatRequest) -> char {
    let mut used = [false; 256];
    let mut mark = |s: &str| {
        for c in s.chars() {
            let u = c as u32;
            if (0xE000..0xE100).contains(&u) {
                used[(u - 0xE000) as usize] = true;
            }
        }
    };
    fn walk(v: &Value, f: &mut dyn FnMut(&str)) {
        match v {
            Value::String(s) => f(s),
            Value::Array(a) => a.iter().for_each(|x| walk(x, f)),
            Value::Object(o) => o.iter().for_each(|(k, x)| {
                f(k);
                walk(x, f)
            }),
            _ => {}
        }
    }
    for m in &chat.messages {
        match &m.content {
            Content::Text(s) => mark(s),
            Content::Parts(p) => p.iter().for_each(|x| {
                if let Part::Text(t) = x {
                    mark(t)
                }
            }),
        }
        if let Some(r) = &m.reasoning {
            mark(r);
        }
        for c in &m.tool_calls {
            mark(&c.id);
            mark(&c.name);
            walk(&c.arguments, &mut mark);
        }
        for s in [&m.tool_call_id, &m.name].into_iter().flatten() {
            mark(s);
        }
    }
    chat.tools.iter().for_each(|t| walk(t, &mut mark));
    walk(&Value::Object(chat.template_kwargs.clone()), &mut mark);
    let free = used.iter().position(|u| !u).unwrap_or(0);
    char::from_u32(0xE000 + free as u32).unwrap_or('\u{E000}')
}

/// `max_tokens` of a request: the requested value, or the model default,
/// within the context left after the prompt.
fn resolve_max_tokens(
    requested: Option<u32>,
    prompt: u32,
    context: u32,
    default: u32,
) -> Result<u32, String> {
    let total = prompt.saturating_add(requested.unwrap_or(0));
    if total > context || prompt >= context {
        return Err(format!(
            "This model's maximum context length is {context} tokens. However, you requested {total} tokens ({prompt} in the messages, {} in the completion). Please reduce the length of the messages or completion.",
            requested.unwrap_or(0)
        ));
    }
    Ok(requested.unwrap_or(default).min(context - prompt).max(1))
}

#[cfg(test)]
#[path = "hf_tests.rs"]
mod tests;

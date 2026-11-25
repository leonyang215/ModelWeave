//! The model registry: every model the tokenizer serves, with its family
//! (prompt format), tokenizer files and settings. One process serves any
//! number of models at once; requests pick one by name or alias.
//!
//! Families:
//!
//! * `deepseek-v4.1`, `deepseek-v4`: deepseek-recipe's prompt encodings;
//! * `qwen3`: Qwen3 text models (ChatML chat template, `enable_thinking`
//!   switch, `<think>` reasoning, hermes `<tool_call>` tool calls);
//! * `qwen3-vl`: Qwen3-VL (the same plus images and videos);
//! * `chat-template`: any model with a Hugging Face chat template; parsers,
//!   stop tokens and the thinking switch come from the configuration.
//!
//! A model is registered from its Hugging Face files: `tokenizer.json`,
//! `tokenizer_config.json` (chat template, special tokens),
//! `generation_config.json` (stop tokens, sampling defaults) and, for
//! vision models, `preprocessor_config.json` / `video_preprocessor_config.json`.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{Value, json};
use tokenizers::Tokenizer;

use crate::deepseek::DeepseekRenderer;
use crate::error::ApiError;
use crate::hf::{TemplateRenderer, TemplateSettings, Thinking};
use crate::metrics;
use crate::protocol::{RenderRequest, RenderResult, Segment, TimingUs};
use crate::template::ContentFormat;
use crate::vision::{Planner, QwenVision, VisionSpec};

/// The prompt format of a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Family {
    DeepseekV41,
    DeepseekV4,
    Qwen3,
    Qwen3Vl,
    ChatTemplate,
}

impl Family {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "deepseek-v4.1" | "dsv41" | "deepseek_v41" => Ok(Self::DeepseekV41),
            "deepseek-v4" | "dsv4" | "deepseek_v4" => Ok(Self::DeepseekV4),
            "qwen3" | "qwen" => Ok(Self::Qwen3),
            "qwen3-vl" | "qwen3_vl" => Ok(Self::Qwen3Vl),
            "chat-template" | "chat_template" | "hf" => Ok(Self::ChatTemplate),
            other => Err(format!(
                "unknown model family {other:?} (want deepseek-v4.1, deepseek-v4, qwen3, qwen3-vl or chat-template)"
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::DeepseekV41 => "deepseek-v4.1",
            Self::DeepseekV4 => "deepseek-v4",
            Self::Qwen3 => "qwen3",
            Self::Qwen3Vl => "qwen3-vl",
            Self::ChatTemplate => "chat-template",
        }
    }

    fn is_deepseek(self) -> bool {
        matches!(self, Self::DeepseekV41 | Self::DeepseekV4)
    }
}

/// Configuration of one model (a `[[models]]` entry of the TOML file).
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    pub name: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    /// Prompt format; `encoding` is accepted as an alias.
    #[serde(alias = "encoding", default = "default_family")]
    pub family: String,
    /// Directory of the model's Hugging Face files; the `*_file` settings
    /// override single files.
    pub path: Option<PathBuf>,
    pub tokenizer_file: Option<PathBuf>,
    /// `tokenizer_config.json`, `chat_template.json` or a `.jinja` file.
    pub chat_template_file: Option<PathBuf>,
    pub generation_config_file: Option<PathBuf>,
    pub preprocessor_config_file: Option<PathBuf>,
    pub video_preprocessor_config_file: Option<PathBuf>,
    #[serde(default = "default_context")]
    pub context_length: u32,
    /// Thinking mode when a request does not choose one.
    pub default_thinking: Option<bool>,
    /// `toggle` (requests choose), `always` or `none`.
    pub thinking: Option<String>,
    /// Chat template variable of the thinking switch (default
    /// `enable_thinking`).
    pub thinking_kwarg: Option<String>,
    /// Output parsers engines apply (empty: none).
    pub reasoning_parser: Option<String>,
    pub tool_call_parser: Option<String>,
    /// Tokens that end generation (default: generation_config.json
    /// `eos_token_id`, else the tokenizer's EOS token).
    pub stop_tokens: Option<Vec<String>>,
    /// How the chat template reads content: `string` or `openai`
    /// (default: detected from the template).
    pub content_format: Option<String>,
    /// `max_tokens` when a request sets none, outside and in thinking mode.
    pub default_max_tokens: Option<u32>,
    pub thinking_max_tokens: Option<u32>,
    #[serde(default)]
    pub vision: VisionOverrides,
}

fn default_family() -> String {
    "deepseek-v4.1".into()
}

fn default_context() -> u32 {
    1_048_576
}

/// Overrides of the vision settings.
#[derive(Debug, Clone, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct VisionOverrides {
    // DeepSeek
    pub video_temporal_patch_size: Option<u32>,
    pub video_default_fps: Option<f64>,
    pub video_frame_max_long_side: Option<u32>,
    // Qwen-VL
    pub patch_size: Option<u32>,
    pub merge_size: Option<u32>,
    pub temporal_patch_size: Option<u32>,
    pub image_min_pixels: Option<u64>,
    pub image_max_pixels: Option<u64>,
    pub video_min_pixels: Option<u64>,
    pub video_max_pixels: Option<u64>,
    pub video_fps: Option<f64>,
    pub video_min_frames: Option<u32>,
    pub video_max_frames: Option<u32>,
    /// Per-frame token ceiling of videos (0: none).
    pub video_max_frame_tokens: Option<u32>,
}

impl VisionOverrides {
    fn apply(&self, spec: &mut VisionSpec) {
        match &mut spec.planner {
            Planner::Deepseek(d) => {
                if let Some(x) = self.video_temporal_patch_size {
                    d.video_temporal_patch_size = x.max(1);
                }
                if let Some(x) = self.video_default_fps {
                    d.video_default_fps = x;
                }
                if let Some(x) = self.video_frame_max_long_side {
                    d.video_frame_max_long_side = x;
                }
            }
            Planner::Qwen(q) => {
                let set = |dst: &mut u32, v: Option<u32>| {
                    if let Some(v) = v {
                        *dst = v.max(1);
                    }
                };
                set(&mut q.patch_size, self.patch_size);
                set(&mut q.merge_size, self.merge_size);
                set(&mut q.temporal_patch_size, self.temporal_patch_size);
                set(&mut q.video_min_frames, self.video_min_frames);
                set(&mut q.video_max_frames, self.video_max_frames);
                for (dst, v) in [
                    (&mut q.image_min_pixels, self.image_min_pixels),
                    (&mut q.image_max_pixels, self.image_max_pixels),
                    (&mut q.video_min_pixels, self.video_min_pixels),
                    (&mut q.video_max_pixels, self.video_max_pixels),
                ] {
                    if let Some(v) = v {
                        *dst = v;
                    }
                }
                if let Some(f) = self.video_fps {
                    q.video_fps = f;
                }
                if let Some(t) = self.video_max_frame_tokens {
                    q.video_max_frame_tokens = (t > 0).then_some(t);
                }
            }
        }
    }
}

/// What every renderer needs to know about its model.
#[derive(Debug, Clone)]
pub struct ModelMeta {
    pub name: String,
    pub family: Family,
    pub context_length: u32,
}

pub enum Renderer {
    Deepseek(Box<DeepseekRenderer>),
    Template(Box<TemplateRenderer>),
}

/// A loaded model.
pub struct Model {
    pub meta: ModelMeta,
    pub aliases: Vec<String>,
    pub tokenizer: Arc<Tokenizer>,
    pub renderer: Renderer,
}

/// The files of a model, resolved from its directory and overrides.
struct Files {
    tokenizer: PathBuf,
    chat_template: Option<PathBuf>,
    generation_config: Option<PathBuf>,
    preprocessor_config: Option<PathBuf>,
    video_preprocessor_config: Option<PathBuf>,
}

impl Files {
    fn resolve(cfg: &ModelConfig) -> Result<Self, String> {
        let dir = cfg.path.as_deref();
        let existing = |name: &str| -> Option<PathBuf> {
            let p = dir?.join(name);
            p.is_file().then_some(p)
        };
        let tokenizer = cfg
            .tokenizer_file
            .clone()
            .or_else(|| dir.map(|d| d.join("tokenizer.json")))
            .ok_or_else(|| format!("model {}: set path or tokenizer_file", cfg.name))?;
        let chat_template = cfg.chat_template_file.clone().or_else(|| {
            existing("chat_template.jinja")
                .or_else(|| existing("chat_template.json"))
                .or_else(|| existing("tokenizer_config.json"))
        });
        Ok(Self {
            tokenizer,
            chat_template,
            generation_config: cfg
                .generation_config_file
                .clone()
                .or_else(|| existing("generation_config.json")),
            preprocessor_config: cfg
                .preprocessor_config_file
                .clone()
                .or_else(|| existing("preprocessor_config.json")),
            video_preprocessor_config: cfg
                .video_preprocessor_config_file
                .clone()
                .or_else(|| existing("video_preprocessor_config.json")),
        })
    }
}

fn read_json(path: &Path) -> Result<Value, String> {
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("parse {}: {e}", path.display()))
}

/// A chat template and the special-token strings of its configuration.
pub struct TemplateSource {
    pub template: String,
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
}

fn token_string(v: Option<&Value>) -> Option<String> {
    match v? {
        Value::String(s) => Some(s.clone()),
        Value::Object(o) => o.get("content")?.as_str().map(str::to_string),
        _ => None,
    }
}

fn load_template(path: &Path) -> Result<TemplateSource, String> {
    if path.extension().and_then(|e| e.to_str()) == Some("jinja") {
        let template =
            std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        return Ok(TemplateSource {
            template,
            bos_token: None,
            eos_token: None,
        });
    }
    let cfg = read_json(path)?;
    let template = match cfg.get("chat_template") {
        Some(Value::String(s)) => s.clone(),
        // A list of named templates: the default one.
        Some(Value::Array(list)) => list
            .iter()
            .find(|t| t.get("name").and_then(Value::as_str) == Some("default"))
            .or_else(|| list.first())
            .and_then(|t| t.get("template"))
            .and_then(Value::as_str)
            .ok_or_else(|| format!("{}: no default chat template", path.display()))?
            .to_string(),
        _ => return Err(format!("{} has no chat_template", path.display())),
    };
    Ok(TemplateSource {
        template,
        bos_token: token_string(cfg.get("bos_token")),
        eos_token: token_string(cfg.get("eos_token")),
    })
}

impl Model {
    pub fn load(cfg: &ModelConfig) -> Result<Self, String> {
        let family = Family::parse(&cfg.family)?;
        let files = Files::resolve(cfg)?;
        let tokenizer = Tokenizer::from_file(&files.tokenizer)
            .map_err(|e| format!("load tokenizer {}: {e}", files.tokenizer.display()))?;
        let meta = ModelMeta {
            name: cfg.name.clone(),
            family,
            context_length: cfg.context_length,
        };
        let renderer = if family.is_deepseek() {
            // V4.1 sees images and videos; V4 is text-only.
            let vision = (family == Family::DeepseekV41).then(|| {
                let mut vision = VisionSpec::deepseek_v41();
                cfg.vision.apply(&mut vision);
                vision
            });
            Renderer::Deepseek(Box::new(DeepseekRenderer::new(
                &tokenizer,
                &meta,
                &cfg.aliases,
                cfg.default_thinking.unwrap_or(true),
                vision,
            )?))
        } else {
            let path = files.chat_template.as_deref().ok_or_else(|| {
                format!(
                    "model {}: no chat template (set chat_template_file or path)",
                    cfg.name
                )
            })?;
            let source = load_template(path)?;
            let generation = files
                .generation_config
                .as_deref()
                .map(read_json)
                .transpose()?;
            let settings = template_settings(cfg, family, &files)?;
            Renderer::Template(Box::new(TemplateRenderer::new(
                &tokenizer,
                source,
                settings,
                generation.as_ref(),
            )?))
        };
        Ok(Self {
            meta,
            aliases: cfg.aliases.clone(),
            tokenizer: Arc::new(tokenizer),
            renderer,
        })
    }

    pub fn matches(&self, model: &str) -> bool {
        self.meta.name == model || self.aliases.iter().any(|a| a == model)
    }

    pub fn render(&self, req: &RenderRequest) -> Result<RenderResult, ApiError> {
        match &self.renderer {
            Renderer::Deepseek(r) => r.render(&self.meta, req),
            Renderer::Template(r) => r.render(&self.meta, req),
        }
    }

    /// Encode text. With `parse_special`, special-token spellings become
    /// special tokens (used by test engines to emit `</think>` or tool-call
    /// markup).
    pub fn encode_text(&self, text: &str, parse_special: bool) -> Result<Vec<u32>, String> {
        if parse_special {
            return self
                .tokenizer
                .encode(text, false)
                .map(|e| e.get_ids().to_vec())
                .map_err(|e| e.to_string());
        }
        match &self.renderer {
            Renderer::Deepseek(r) => r.encode_plain(text),
            Renderer::Template(r) => r.encode_plain(text),
        }
    }

    pub fn decode_text(&self, ids: &[u32], skip_special: bool) -> Result<String, String> {
        self.tokenizer
            .decode(ids, skip_special)
            .map_err(|e| e.to_string())
    }

    /// Decode every token on its own (engines stream such pieces).
    pub fn decode_pieces(&self, ids: &[u32], skip_special: bool) -> Result<Vec<String>, String> {
        ids.iter()
            .map(|id| {
                self.tokenizer
                    .decode(&[*id], skip_special)
                    .map_err(|e| e.to_string())
            })
            .collect()
    }

    /// What `GET /v1/models` reports.
    pub fn info(&self) -> Value {
        let mut v = json!({
            "name": self.meta.name,
            "aliases": self.aliases,
            "family": self.meta.family.name(),
            "encoding": self.meta.family.name(),
            "context_length": self.meta.context_length,
        });
        let (stop, reasoning, tools, thinking, default_thinking, think, modalities) =
            match &self.renderer {
                Renderer::Deepseek(r) => (
                    r.eos_ids.clone(),
                    Some(crate::deepseek::PARSER.to_string()),
                    Some(crate::deepseek::PARSER.to_string()),
                    "toggle",
                    None,
                    r.think_ids,
                    if r.vision.is_some() {
                        vec!["text", "image", "video"]
                    } else {
                        vec!["text"]
                    },
                ),
                Renderer::Template(r) => {
                    let s = &r.settings;
                    let mut modalities = vec!["text"];
                    if s.vision.is_some() {
                        modalities.extend(["image", "video"]);
                    }
                    (
                        r.stop_ids.clone(),
                        s.reasoning_parser.clone(),
                        s.tool_call_parser.clone(),
                        s.thinking.name(),
                        (s.thinking == Thinking::Toggle).then_some(s.default_thinking),
                        r.think_ids,
                        modalities,
                    )
                }
            };
        v["eos_token_ids"] = json!(stop);
        v["reasoning_parser"] = json!(reasoning);
        v["tool_call_parser"] = json!(tools);
        v["thinking"] = json!(thinking);
        if let Some(d) = default_thinking {
            v["default_thinking"] = json!(d);
        }
        v["think_start_token_id"] = json!(think.0);
        v["think_end_token_id"] = json!(think.1);
        v["modalities"] = json!(modalities);
        v
    }
}

/// Settings of a chat-template model: family defaults, then the
/// configuration.
fn template_settings(
    cfg: &ModelConfig,
    family: Family,
    files: &Files,
) -> Result<TemplateSettings, String> {
    let mut s = TemplateSettings::for_family(family);
    if let Some(t) = &cfg.thinking {
        s.thinking = Thinking::parse(t)?;
        if s.thinking != Thinking::None && s.reasoning_parser.is_none() {
            s.reasoning_parser = Some("qwen3".into());
        }
    }
    if let Some(d) = cfg.default_thinking {
        s.default_thinking = d;
    }
    if let Some(k) = &cfg.thinking_kwarg {
        s.thinking_kwarg = k.clone();
    }
    let parser = |v: &String| (!v.is_empty()).then(|| v.clone());
    if let Some(p) = &cfg.reasoning_parser {
        s.reasoning_parser = parser(p);
    }
    if let Some(p) = &cfg.tool_call_parser {
        s.tool_call_parser = parser(p);
    }
    if let Some(t) = &cfg.stop_tokens {
        s.stop_tokens = t.clone();
    }
    if let Some(f) = &cfg.content_format {
        s.content_format = Some(ContentFormat::parse(f)?);
    }
    if let Some(n) = cfg.default_max_tokens {
        s.default_max_tokens = n;
    }
    if let Some(n) = cfg.thinking_max_tokens {
        s.thinking_max_tokens = n;
    }
    if let Some(vision) = &mut s.vision {
        if let Planner::Qwen(q) = &mut vision.planner {
            if let Some(p) = &files.preprocessor_config {
                q.apply_processor_config(&read_json(p)?, false);
            }
            if let Some(p) = &files.video_preprocessor_config {
                q.apply_processor_config(&read_json(p)?, true);
            }
        }
        cfg.vision.apply(vision);
    }
    Ok(s)
}

impl TemplateSettings {
    fn for_family(family: Family) -> Self {
        let qwen = |vision: Option<VisionSpec>| Self {
            thinking: Thinking::None,
            default_thinking: true,
            thinking_kwarg: "enable_thinking".into(),
            reasoning_parser: None,
            tool_call_parser: Some("hermes".into()),
            stop_tokens: Vec::new(),
            content_format: None,
            default_max_tokens: 8192,
            thinking_max_tokens: 32768,
            role_marker: Some("<|im_start|>".into()),
            think_tokens: Some(("<think>".into(), "</think>".into())),
            vision,
        };
        match family {
            Family::Qwen3 => Self {
                thinking: Thinking::Toggle,
                reasoning_parser: Some("qwen3".into()),
                ..qwen(None)
            },
            Family::Qwen3Vl => Self {
                default_max_tokens: 16384,
                ..qwen(Some(VisionSpec::qwen(QwenVision::qwen3_vl())))
            },
            _ => Self {
                thinking: Thinking::None,
                default_thinking: false,
                thinking_kwarg: "enable_thinking".into(),
                reasoning_parser: None,
                tool_call_parser: None,
                stop_tokens: Vec::new(),
                content_format: None,
                default_max_tokens: 8192,
                thinking_max_tokens: 32768,
                role_marker: None,
                think_tokens: None,
                vision: None,
            },
        }
    }
}

/// Load every configured model; names and aliases must be unique.
pub fn load_all(configs: &[ModelConfig]) -> Result<Vec<Arc<Model>>, String> {
    let mut models: Vec<Arc<Model>> = Vec::new();
    for cfg in configs {
        for n in std::iter::once(&cfg.name).chain(&cfg.aliases) {
            if models.iter().any(|m| m.matches(n)) {
                return Err(format!("model name {n:?} is used twice"));
            }
        }
        models.push(Arc::new(Model::load(cfg)?));
    }
    Ok(models)
}

// ---------------------------------------------------------------------------
// Shared helpers of the renderers
// ---------------------------------------------------------------------------

/// Times the render stages and records them.
pub struct Stopwatch {
    t0: Instant,
    last: Duration,
    laps: Vec<(&'static str, Duration)>,
}

impl Stopwatch {
    pub fn new() -> Self {
        Self {
            t0: Instant::now(),
            last: Duration::ZERO,
            laps: Vec::with_capacity(5),
        }
    }

    /// Close the stage that ends now.
    pub fn lap(&mut self, stage: &'static str) {
        let now = self.t0.elapsed();
        self.laps.push((stage, now - self.last));
        self.last = now;
    }

    /// Record the stages and the prompt length; returns the timings.
    pub fn finish(&self, model: &str, protocol: &str, prompt_tokens: usize) -> TimingUs {
        let total = self.t0.elapsed();
        let mut t = TimingUs {
            total: total.as_micros() as u64,
            ..TimingUs::default()
        };
        for (stage, d) in &self.laps {
            let us = d.as_micros() as u64;
            match *stage {
                "parse" => t.parse = us,
                "convert" => t.convert = us,
                "render" => t.render = us,
                "tokenize" => t.tokenize = us,
                "media" => t.media = us,
                _ => {}
            }
            metrics::stage(model, protocol, stage, *d);
        }
        metrics::stage(model, protocol, "total", total);
        metrics::prompt_tokens(model, protocol, prompt_tokens);
        t
    }
}

impl Default for Stopwatch {
    fn default() -> Self {
        Self::new()
    }
}

/// Token ranges of the rendered messages, split at role tokens.
pub fn segments_by_role_token(ids: &[u32], role_tokens: &[(u32, &'static str)]) -> Vec<Segment> {
    let mut out: Vec<Segment> = Vec::new();
    for (i, t) in ids.iter().enumerate() {
        if let Some((_, role)) = role_tokens.iter().find(|(id, _)| id == t) {
            open_segment(&mut out, i, role);
        }
    }
    close_segments(&mut out, ids.len());
    out
}

/// Token ranges of the rendered messages, split at a role marker followed
/// by a role name token (ChatML `<|im_start|>user`).
pub fn segments_by_marker(ids: &[u32], marker: u32, roles: &[(u32, &'static str)]) -> Vec<Segment> {
    let mut out: Vec<Segment> = Vec::new();
    for (i, w) in ids.windows(2).enumerate() {
        if w[0] == marker
            && let Some((_, role)) = roles.iter().find(|(id, _)| *id == w[1])
        {
            open_segment(&mut out, i, role);
        }
    }
    close_segments(&mut out, ids.len());
    out
}

fn open_segment(out: &mut Vec<Segment>, at: usize, role: &str) {
    if let Some(last) = out.last_mut() {
        last.length = at - last.offset;
    }
    out.push(Segment {
        role: role.into(),
        offset: at,
        length: 0,
    });
}

fn close_segments(out: &mut [Segment], n: usize) {
    if let Some(last) = out.last_mut() {
        last.length = n - last.offset;
    }
}

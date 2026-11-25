//! Wire types of the tokenizer (docs/protocol.md).
//!
//! The tokenizer converts and tokenizes, nothing else. Its result is the
//! intermediate state: the conversation rendered into one token list by the
//! model's prompt format, with the options engines need to generate and
//! parse (`conversation`, `inference_options`, `parsing_options`, `stream`,
//! `response_options`). Images and videos are already expanded in that list:
//! every position of an item holds the item's `pad_value`, an
//! out-of-vocabulary token derived from the media digest (as SGLang pads
//! multimodal prompts), and the number of positions is computed from the
//! media metadata and pixel controls the caller sends.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use serde_json::value::RawValue;

pub const PROTOCOL_CHAT_COMPLETIONS: &str = "chat_completions";
pub const PROTOCOL_RESPONSES: &str = "responses";
pub const PROTOCOL_MESSAGES: &str = "messages";

/// Render request (POST /v1/render).
#[derive(Debug, Deserialize)]
pub struct RenderRequest {
    #[serde(default)]
    pub request_id: String,
    pub protocol: String,
    /// Tokenizer-side model name; defaults to `body.model`.
    #[serde(default)]
    pub model: Option<String>,
    pub body: Box<RawValue>,
    #[serde(default)]
    pub context_length: Option<u32>,
}

// ---------------------------------------------------------------------------
// Media inputs (ModelWeave extensions on content blocks)
// ---------------------------------------------------------------------------

/// Objective properties of an image or a video, measured by the caller. The
/// tokenizer never fetches or decodes remote media.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct MediaMetadata {
    /// sha256 lowercase hex of the media bytes (64 chars).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub digest: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub duration_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_frames: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub video_fps: Option<f64>,
}

/// Pixel cap of one image or one video frame. Unset fields take the media
/// type's default.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PixelControlOptions {
    /// Longest side in pixels before the model's own resize.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_long_side: Option<u32>,
    /// Pixel budget after the resize: width × height for an image,
    /// width × height × kept frames for a video.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_total_pixels: Option<u64>,
}

/// Frame sampling of a video: a target sampling rate or a fixed frame count,
/// and the per-frame pixel cap.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct FrameExtractOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fps: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nframes: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pixel_control_options: Option<PixelControlOptions>,
}

// ---------------------------------------------------------------------------
// The intermediate state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub struct Span {
    pub offset: usize,
    pub length: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct MediaSource {
    /// `url` or `base64` (a data URL).
    pub kind: &'static str,
    pub url: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub mime: String,
}

/// The result of planning one item: the size the model's preprocessing
/// produces and the tokens per placement. Videos add the frame sampling.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct MediaPlan {
    /// Width and height after the resize (per frame for a video).
    pub width: u32,
    pub height: u32,
    /// Tokens of the image, or of one kept frame.
    pub tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nframes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kept_frames: Option<usize>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub timestamps: Vec<f64>,
}

/// The token range of one image or one kept video frame.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Placement {
    pub offset: usize,
    pub length: usize,
    /// The token every position of the range holds.
    pub pad_value: u32,
    /// What the engine scatters here: the image digest, or
    /// `{video digest}_kf_{k:04}` for kept frame k.
    pub reference: String,
    /// `image` or `frame`.
    pub modality: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MediaItem {
    pub index: usize,
    /// `image` or `video`.
    pub modality: &'static str,
    pub digest: String,
    pub source: MediaSource,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    pub metadata: MediaMetadata,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pixel_control_options: Option<PixelControlOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frame_extract_options: Option<FrameExtractOptions>,
    pub plan: MediaPlan,
    /// The whole range of the item (a video's includes its timestamp text).
    pub span: Span,
    pub placements: Vec<Placement>,
    /// The tokens that replace `span` for engines that expand media
    /// themselves (the model's placeholder); absent when the model has none
    /// for this modality, so only engines taking the expanded list can run it.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub placeholder: Vec<u32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Multimodal {
    /// Every pad_value is at least this, far outside the vocabulary.
    pub pad_shift: u32,
    /// The model token one unexpanded image occupies in the prompt format,
    /// for engines that expand media themselves.
    #[serde(skip_serializing_if = "is_zero")]
    pub placeholder_token_id: u32,
    pub items: Vec<MediaItem>,
}

fn is_zero(v: &u32) -> bool {
    *v == 0
}

/// The conversation without its messages (they are in `token_ids` now):
/// what engine-side parsers need.
#[derive(Debug, Clone, Serialize)]
pub struct ConversationInfo {
    pub thinking_mode: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    /// OpenAI function tools in declaration order (omitted with tool choice
    /// `none`).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<Value>,
    /// `auto`, `none` or `required`.
    pub tool_choice: &'static str,
    /// `text`, `json_object`, `json_schema` or `regex`.
    pub response_format: &'static str,
    /// The chat template variables the prompt was rendered with (thinking
    /// switches); engines that parse their output with the template's
    /// parsers take them as is.
    #[serde(skip_serializing_if = "serde_json::Map::is_empty")]
    pub chat_template_kwargs: serde_json::Map<String, Value>,
}

/// A constraint engines may enforce by constrained decoding, as kind and
/// constraint string.
#[derive(Debug, Clone, Serialize)]
pub struct OutputConstraint {
    /// `json`, `regex`, `tool_call` or `tool_call_required`.
    pub kind: &'static str,
    pub schema: String,
}

/// Sampling options, with `max_tokens` resolved and the stop token IDs of the
/// model.
#[derive(Debug, Clone, Serialize)]
pub struct InferenceOptions {
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub min_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    pub logprobs: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u8>,
    pub stop_token_ids: Vec<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_constraint: Option<OutputConstraint>,
}

/// How engines parse the output.
#[derive(Debug, Clone, Serialize)]
pub struct ParsingOptions {
    /// The output format of reasoning (`deepseek_v4`, `qwen3`); absent for
    /// models without reasoning.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_parser: Option<String>,
    /// The output format of tool calls (`deepseek_v4` DSML, `hermes`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_parser: Option<String>,
    pub parse_tool_calls: bool,
    pub starts_in_tool_calls: bool,
    pub parse_json_output: bool,
    /// `reasoning_start`, `reasoning_continuation` or
    /// `content_continuation`; absent outside thinking mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_initial_stage: Option<&'static str>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub stop_sequences: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResponseOptions {
    pub include_usage: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct TokenLayout {
    /// Prefix that stays identical when the conversation continues.
    pub cacheable_prefix: Span,
    /// Trailing prompt tokens that open the model's turn (DeepSeek V4.1
    /// `<think>` or `</think>`, ChatML `<|im_start|>assistant\n`).
    pub generation_primer: Span,
}

#[derive(Debug, Clone, Serialize)]
pub struct Segment {
    pub role: String,
    pub offset: usize,
    pub length: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct RenderArtifact {
    pub version: u32,
    pub encoding: String,
    pub token_layout: TokenLayout,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub segments: Vec<Segment>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub prefix_continuation: bool,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct TimingUs {
    pub parse: u64,
    pub convert: u64,
    pub render: u64,
    pub tokenize: u64,
    pub media: u64,
    pub total: u64,
}

/// Result of POST /v1/render: the intermediate state.
#[derive(Debug, Clone, Serialize)]
pub struct RenderResult {
    pub request_id: String,
    /// The client API protocol the gateway answers in.
    pub protocol: String,
    pub model: String,
    /// The model family whose prompt format rendered the request.
    pub model_version: String,
    /// Length of `token_ids`: the prompt as the engine lays it out.
    pub prompt_tokens: usize,
    pub token_ids: Vec<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub multimodal: Option<Multimodal>,
    pub conversation: ConversationInfo,
    pub inference_options: InferenceOptions,
    pub parsing_options: ParsingOptions,
    pub stream: bool,
    pub response_options: ResponseOptions,
    pub render_artifact: RenderArtifact,
    pub timing_us: TimingUs,
}

//! DeepSeek V4 / V4.1 renderer built on deepseek-recipe: API request ->
//! intermediate state (recipe's ConversationRequest with the conversation
//! rendered into one token list, media expanded to pad values).
//!
//! The tokenizer only converts and tokenizes. Engines parse their output and
//! the gateway formats the client response.

use deepseek_recipe::anthropic::MessagesRequest;
use deepseek_recipe::openai::{ChatCompletionRequest, ResponsesRequest};
use deepseek_recipe::request::{ConversionOptions, ModelVersion, ProtocolRequest};
use deepseek_recipe::stream::ThinkingInitialStage;
use deepseek_recipe_core::conversation::{ReasoningEffort, ResponseFormat};
use deepseek_recipe_core::messages::InputMessage;
use deepseek_recipe_core::multimodal::{IMAGE_SPECIAL_TOKEN_ID, ImageSource};
use deepseek_recipe_core::special_tokens::SpecialToken;
use deepseek_recipe_core::tools::ToolChoice;
use deepseek_recipe_encoding::v4::dsv4::DeepseekV4Encoding;
use deepseek_recipe_encoding::v4::dsv41::DeepseekV41Encoding;
use deepseek_recipe_encoding::{PromptEncoding, PromptTokenizer, RenderedPrompt, TokenizerEncoder};
use serde::de::DeserializeOwned;
use serde_json::{Map, Value, json};
use tokenizers::Tokenizer;

use crate::error::ApiError;
use crate::media::{self, MediaLayout, Slot};
use crate::metrics;
use crate::model::{Family, ModelMeta, Stopwatch, segments_by_role_token};
use crate::protocol::*;
use crate::vision::VisionSpec;

/// Parser names engines use for DeepSeek V4 / V4.1 output (`</think>`
/// reasoning, DSML tool calls).
pub const PARSER: &str = "deepseek_v4";

enum Encoding {
    V41(DeepseekV41Encoding),
    V4(DeepseekV4Encoding),
}

impl Encoding {
    fn render(
        &self,
        conversation: &deepseek_recipe_core::conversation::Conversation,
    ) -> RenderedPrompt {
        match self {
            Self::V41(e) => e.render_conversation(conversation),
            Self::V4(e) => e.render_conversation(conversation),
        }
    }
}

/// Prompt renderer of one DeepSeek model.
pub struct DeepseekRenderer {
    encoding: Encoding,
    prompt_tokenizer: PromptTokenizer,
    options: ConversionOptions,
    pub eos_ids: Vec<u32>,
    pub think_ids: (Option<u32>, Option<u32>),
    role_tokens: Vec<(u32, &'static str)>,
    /// Image and video planning; `None` for text-only V4.
    pub vision: Option<VisionSpec>,
}

/// Lets the generic render path override the model name of every protocol.
pub trait HasModel {
    fn model_mut(&mut self) -> &mut String;
}

impl HasModel for ChatCompletionRequest {
    fn model_mut(&mut self) -> &mut String {
        &mut self.model
    }
}
impl HasModel for ResponsesRequest {
    fn model_mut(&mut self) -> &mut String {
        &mut self.model
    }
}
impl HasModel for MessagesRequest {
    fn model_mut(&mut self) -> &mut String {
        &mut self.model
    }
}

impl DeepseekRenderer {
    pub fn new(
        tokenizer: &Tokenizer,
        meta: &ModelMeta,
        aliases: &[String],
        default_thinking: bool,
        vision: Option<VisionSpec>,
    ) -> Result<Self, String> {
        let token_id = |t: SpecialToken| tokenizer.token_to_id(t.default_str());
        let eos = token_id(SpecialToken::Eos)
            .ok_or_else(|| "tokenizer has no end-of-sentence token".to_string())?;
        let mut role_tokens = Vec::new();
        for (tok, role) in [
            (SpecialToken::System, "system"),
            (SpecialToken::User, "user"),
            (SpecialToken::Assistant, "assistant"),
        ] {
            if let Some(id) = token_id(tok) {
                role_tokens.push((id, role));
            }
        }
        let (encoding, version) = match meta.family {
            Family::DeepseekV4 => (
                Encoding::V4(DeepseekV4Encoding::new()),
                ModelVersion::DeepseekV4Pro,
            ),
            _ => (
                Encoding::V41(DeepseekV41Encoding::new()),
                ModelVersion::DeepseekV41Flash,
            ),
        };
        let mut options = ConversionOptions::new()
            .with_default_thinking_mode(default_thinking)
            .with_model_alias(meta.name.clone(), version);
        for alias in aliases {
            options = options.with_model_alias(alias.clone(), version);
        }
        Ok(Self {
            encoding,
            prompt_tokenizer: PromptTokenizer::new(tokenizer),
            options,
            eos_ids: vec![eos],
            think_ids: (
                token_id(SpecialToken::ThinkingStart),
                token_id(SpecialToken::ThinkingEnd),
            ),
            role_tokens,
            vision,
        })
    }

    /// Render a request of any supported protocol.
    pub fn render(&self, meta: &ModelMeta, req: &RenderRequest) -> Result<RenderResult, ApiError> {
        match req.protocol.as_str() {
            PROTOCOL_CHAT_COMPLETIONS => self.render_as::<ChatCompletionRequest>(meta, req),
            PROTOCOL_RESPONSES => self.render_as::<ResponsesRequest>(meta, req),
            PROTOCOL_MESSAGES => self.render_as::<MessagesRequest>(meta, req),
            other => Err(ApiError::bad_request(
                &req.protocol,
                format!(
                    "unknown protocol {other:?} (want chat_completions, responses or messages)"
                ),
            )),
        }
    }

    fn render_as<T>(&self, meta: &ModelMeta, req: &RenderRequest) -> Result<RenderResult, ApiError>
    where
        T: ProtocolRequest + DeserializeOwned + HasModel,
    {
        let protocol = req.protocol.as_str();
        let bad = |stage: &str, msg: String| {
            metrics::error(&meta.name, stage);
            ApiError::bad_request(protocol, msg)
        };
        let mut sw = Stopwatch::new();
        let mut body: Value = serde_json::from_str(req.body.get())
            .map_err(|e| bad("parse", format!("invalid JSON request: {e}")))?;
        let media = media::extract(protocol, &mut body).map_err(|e| bad("media", e))?;
        // Recipe would render V4's images as "[Unsupported Image]" text: a
        // text-only model rejects media instead, like every other one.
        if !media.is_empty() && self.vision.is_none() {
            return Err(bad(
                "media",
                format!("model {} does not accept images or videos", meta.name),
            ));
        }
        let mut parsed: T = serde_json::from_value(body)
            .map_err(|e| bad("parse", format!("invalid JSON request: {e}")))?;
        *parsed.model_mut() = meta.name.clone();
        sw.lap("parse");

        let conv = parsed.convert(self.options.clone()).map_err(|e| {
            metrics::error(&meta.name, "convert");
            ApiError::from_conversion(protocol, e)
        })?;
        sw.lap("convert");

        let rendered = self.encoding.render(&conv.conversation);
        sw.lap("render");
        let raw_ids = self
            .prompt_tokenizer
            .encode_ids(&rendered.prompt, &rendered.special_ranges)
            .map_err(|e| {
                metrics::error(&meta.name, "tokenize");
                ApiError::internal(protocol, format!("tokenize: {e}"))
            })?;
        sw.lap("tokenize");

        // Plan every image and video, then expand their placeholders.
        let planned = match &self.vision {
            Some(v) => media::plan_all(v, &media).map_err(|e| bad("media", e))?,
            None => Vec::new(),
        };
        let slots = self
            .slots(&raw_ids, &rendered.image_sources, planned.len())
            .map_err(|e| {
                metrics::error(&meta.name, "media");
                ApiError::internal(protocol, e)
            })?;
        let layout = MediaLayout {
            image_placeholder: vec![IMAGE_SPECIAL_TOKEN_ID],
            // V4.1 has no video placeholder for engines that expand media.
            video_placeholder: Vec::new(),
            frame_text: |t| format!("{t:.1} seconds"),
            frame_open: None,
            frame_close: None,
        };
        let encode = |text: &str| self.prompt_tokenizer.encode_ids(text, &[]);
        let (token_ids, items) = media::expand(&raw_ids, &slots, &planned, &layout, &encode)
            .map_err(|e| {
                metrics::error(&meta.name, "media");
                ApiError::internal(protocol, e)
            })?;
        sw.lap("media");
        for p in &planned {
            metrics::media(&meta.name, p.input.modality.name());
        }

        let prompt_tokens = token_ids.len();
        let context_limit = req.context_length.unwrap_or(meta.context_length);
        let max_tokens = conv
            .resolve_max_tokens(prompt_tokens as u32, context_limit)
            .map_err(|e| ApiError::from_conversion(protocol, e))?;

        let conversation = &conv.conversation;
        let tool_choice = match conversation.tool_choice {
            ToolChoice::Auto => "auto",
            ToolChoice::None => "none",
            ToolChoice::Required => "required",
        };
        let tools: Vec<Value> = conversation
            .tools
            .iter()
            .filter(|_| tool_choice != "none")
            .map(|t| {
                let mut f = Map::new();
                f.insert("name".into(), json!(t.name));
                if let Some(d) = &t.description {
                    f.insert("description".into(), json!(d));
                }
                f.insert("parameters".into(), t.parameters.clone());
                if let Some(s) = t.strict {
                    f.insert("strict".into(), json!(s));
                }
                json!({"type": "function", "function": f})
            })
            .collect();
        let response_format = match &conversation.response_format {
            ResponseFormat::Text => "text",
            ResponseFormat::JsonObject => "json_object",
            ResponseFormat::JsonSchema(_) => "json_schema",
            ResponseFormat::Regex(_) => "regex",
        };
        let io = &conv.inference_options;
        let po = &conv.parsing_options;
        let segments = segments_by_role_token(&token_ids, &self.role_tokens);
        let n = token_ids.len();
        // The generation primer is what follows the last assistant role token
        // when the model's turn has been opened: `<think>` / `</think>`, plus
        // the DSML opener for required tool calls, or the continued prefix.
        let primer = match segments.last() {
            Some(seg) if seg.role == "assistant" => n - seg.offset - 1,
            _ => 0,
        };
        let prefix_continuation = matches!(
            conversation.messages.last(),
            Some(InputMessage::Assistant { prefix: true, .. })
        );
        let reasoning_effort = conversation.reasoning_effort.map(effort_name);
        let mut kwargs = Map::new();
        kwargs.insert("thinking".into(), json!(conversation.thinking_mode));
        if let Some(e) = &reasoning_effort {
            kwargs.insert("reasoning_effort".into(), json!(e));
        }
        let multimodal = (!items.is_empty()).then_some(Multimodal {
            pad_shift: media::PAD_SHIFT,
            placeholder_token_id: IMAGE_SPECIAL_TOKEN_ID,
            items,
        });

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
                thinking_mode: conversation.thinking_mode,
                reasoning_effort,
                tools,
                tool_choice,
                response_format,
                chat_template_kwargs: kwargs,
            },
            inference_options: InferenceOptions {
                max_tokens,
                temperature: io.temperature,
                top_p: io.top_p,
                top_k: None,
                min_p: None,
                presence_penalty: None,
                frequency_penalty: None,
                repetition_penalty: None,
                seed: io.seed,
                logprobs: io.logprobs,
                top_logprobs: io.top_logprobs,
                stop_token_ids: self.eos_ids.clone(),
                output_constraint: io.output_constraint.as_ref().map(|c| OutputConstraint {
                    kind: c.kind(),
                    schema: c.schema(),
                }),
            },
            parsing_options: ParsingOptions {
                reasoning_parser: Some(PARSER.into()),
                tool_call_parser: Some(PARSER.into()),
                parse_tool_calls: po.parse_tool_calls,
                starts_in_tool_calls: po.starts_in_tool_calls,
                parse_json_output: po.parse_json_output,
                thinking_initial_stage: po.thinking_initial_stage.map(stage_name),
                stop_sequences: po.stop_sequences.clone(),
            },
            stream: conv.stream,
            response_options: ResponseOptions {
                include_usage: conv.response_options.include_usage,
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
                prefix_continuation,
            },
            timing_us: timing,
        })
    }

    /// Map every rendered image placeholder to its media item through the
    /// synthetic source URL of the extraction.
    fn slots(
        &self,
        ids: &[u32],
        sources: &[ImageSource],
        items: usize,
    ) -> Result<Vec<Slot>, String> {
        let positions: Vec<usize> = ids
            .iter()
            .enumerate()
            .filter(|(_, t)| **t == IMAGE_SPECIAL_TOKEN_ID)
            .map(|(i, _)| i)
            .collect();
        if positions.len() != sources.len() || sources.len() != items {
            return Err(format!(
                "found {} image placeholders for {} rendered sources and {items} media items",
                positions.len(),
                sources.len(),
            ));
        }
        positions
            .iter()
            .zip(sources)
            .map(|(&start, src)| {
                let ImageSource::Url { url, .. } = src else {
                    return Err("a rendered image did not come from the media extraction".into());
                };
                let media = media::synthetic_index(url)
                    .filter(|i| *i < items)
                    .ok_or_else(|| format!("unexpected rendered image source {url}"))?;
                Ok(Slot {
                    start,
                    len: 1,
                    media,
                })
            })
            .collect()
    }

    /// Encode text without special tokens (plain text) or with them.
    pub fn encode_plain(&self, text: &str) -> Result<Vec<u32>, String> {
        self.prompt_tokenizer.encode_ids(text, &[])
    }
}

fn effort_name(e: ReasoningEffort) -> String {
    match e {
        ReasoningEffort::Low => "low",
        ReasoningEffort::High => "high",
        ReasoningEffort::Xhigh => "xhigh",
        ReasoningEffort::Max => "max",
    }
    .to_string()
}

fn stage_name(s: ThinkingInitialStage) -> &'static str {
    match s {
        ThinkingInitialStage::ReasoningStart => "reasoning_start",
        ThinkingInitialStage::ReasoningContinuation => "reasoning_continuation",
        ThinkingInitialStage::ContentContinuation => "content_continuation",
    }
}

#[cfg(test)]
#[path = "deepseek_tests.rs"]
mod tests;

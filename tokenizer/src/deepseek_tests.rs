use std::sync::{Arc, LazyLock};

use base64::Engine as _;
use deepseek_recipe_core::multimodal::ImageTokenSpec;
use serde_json::{Value, json};

use super::*;
use crate::media::{kept_frame_reference, pad_value, sha256_hex};
use crate::model::{Model, ModelConfig};
use crate::vision::limit_long_side;

static CODEC: LazyLock<Arc<Model>> = LazyLock::new(|| {
    let path = format!(
        "{}/../assets/tokenizers/deepseek-v4.1/tokenizer.json",
        env!("CARGO_MANIFEST_DIR")
    );
    Arc::new(
        Model::load(&ModelConfig {
            name: "DeepSeek-V4.1-Flash".into(),
            aliases: vec!["deepseek-v4.1".into()],
            family: "deepseek-v4.1".into(),
            tokenizer_file: Some(path.into()),
            context_length: 1_048_576,
            default_thinking: Some(true),
            ..ModelConfig::default()
        })
        .expect("load model"),
    )
});

fn timestamp_text(t: f64) -> String {
    format!("{t:.1} seconds")
}

fn request(protocol: &str, body: Value) -> RenderRequest {
    RenderRequest {
        request_id: "req-1".into(),
        protocol: protocol.into(),
        model: None,
        body: serde_json::value::RawValue::from_string(body.to_string()).unwrap(),
        context_length: None,
    }
}

fn render(protocol: &str, body: Value) -> RenderResult {
    CODEC.render(&request(protocol, body)).expect("render")
}

fn render_err(protocol: &str, body: Value) -> String {
    CODEC
        .render(&request(protocol, body))
        .expect_err("must fail")
        .body()
}

const BOS: u32 = 0;
const EOS: u32 = 1;
const THINK: u32 = 128821;
const THINK_END: u32 = 128822;

fn keys(v: &Value) -> Vec<&str> {
    v.as_object().unwrap().keys().map(|k| k.as_str()).collect()
}

#[test]
fn chat_render_non_thinking() {
    let r = render(
        PROTOCOL_CHAT_COMPLETIONS,
        json!({
            "model": "anything-the-gateway-serves",
            "messages": [
                {"role": "system", "content": "You are terse."},
                {"role": "user", "content": "Hello"}
            ],
            "thinking": {"type": "disabled"},
            "temperature": 0.3,
            "stop": ["END"]
        }),
    );
    assert_eq!(r.token_ids.first(), Some(&BOS));
    assert_eq!(r.token_ids.last(), Some(&THINK_END));
    assert_eq!(r.prompt_tokens, r.token_ids.len());
    assert!(r.multimodal.is_none());
    assert!(!r.conversation.thinking_mode && !r.stream);
    let io = &r.inference_options;
    assert_eq!(io.max_tokens, 8192);
    assert_eq!(io.temperature, Some(0.3));
    assert_eq!(io.stop_token_ids, vec![EOS]);
    assert_eq!(r.parsing_options.stop_sequences, vec!["END".to_string()]);
    assert_eq!(r.parsing_options.thinking_initial_stage, None);
    assert_eq!(r.render_artifact.token_layout.generation_primer.length, 1);
    let roles: Vec<&str> = r
        .render_artifact
        .segments
        .iter()
        .map(|s| s.role.as_str())
        .collect();
    assert_eq!(roles, vec!["system", "user", "assistant"]);
}

/// The intermediate state mirrors deepseek-recipe's ConversationRequest.
#[test]
fn intermediate_state_mirrors_the_recipe_structure() {
    let r = render(
        PROTOCOL_CHAT_COMPLETIONS,
        json!({"model": "x", "messages": [{"role": "user", "content": "hi"}], "seed": 7,
               "logprobs": true, "top_logprobs": 3, "stream": true,
               "stream_options": {"include_usage": true}}),
    );
    let v = serde_json::to_value(&r).unwrap();
    assert_eq!(
        keys(&v),
        vec![
            "request_id",
            "protocol",
            "model",
            "model_version",
            "prompt_tokens",
            "token_ids",
            "conversation",
            "inference_options",
            "parsing_options",
            "stream",
            "response_options",
            "render_artifact",
            "timing_us"
        ]
    );
    assert_eq!(
        keys(&v["inference_options"]),
        vec![
            "max_tokens",
            "temperature",
            "top_p",
            "seed",
            "logprobs",
            "top_logprobs",
            "stop_token_ids"
        ]
    );
    assert_eq!(
        keys(&v["parsing_options"]),
        vec![
            "reasoning_parser",
            "tool_call_parser",
            "parse_tool_calls",
            "starts_in_tool_calls",
            "parse_json_output",
            "thinking_initial_stage"
        ]
    );
    assert_eq!(v["inference_options"]["seed"], 7);
    assert_eq!(v["inference_options"]["top_logprobs"], 3);
    assert_eq!(
        v["parsing_options"]["thinking_initial_stage"],
        "reasoning_start"
    );
    assert_eq!(v["conversation"]["reasoning_effort"], "high");
    assert_eq!(
        v["conversation"]["chat_template_kwargs"],
        json!({"thinking": true, "reasoning_effort": "high"})
    );
    assert_eq!(v["parsing_options"]["reasoning_parser"], "deepseek_v4");
    assert_eq!(v["model_version"], "deepseek-v4.1");
    assert_eq!(v["response_options"]["include_usage"], true);
}

#[test]
fn chat_render_thinking_default_and_effort() {
    let r = render(
        PROTOCOL_CHAT_COMPLETIONS,
        json!({"model": "x", "messages": [{"role": "user", "content": "2+2?"}], "stream": true}),
    );
    assert_eq!(r.token_ids.last(), Some(&THINK));
    assert!(r.conversation.thinking_mode);
    assert_eq!(r.conversation.reasoning_effort.as_deref(), Some("high"));
    assert_eq!(r.inference_options.max_tokens, 65536);
    assert_eq!(r.inference_options.temperature, Some(1.0));
}

#[test]
fn messages_and_responses_render() {
    let m = render(
        PROTOCOL_MESSAGES,
        json!({"model": "x", "max_tokens": 100, "messages": [{"role": "user", "content": "hi"}],
               "stop_sequences": ["STOP"]}),
    );
    assert_eq!(m.inference_options.max_tokens, 100);
    assert_eq!(m.protocol, PROTOCOL_MESSAGES);
    assert_eq!(m.parsing_options.stop_sequences, vec!["STOP".to_string()]);
    let r = render(
        PROTOCOL_RESPONSES,
        json!({"model": "x", "input": "hi", "max_output_tokens": 50}),
    );
    assert_eq!(r.inference_options.max_tokens, 50);
    assert_eq!(r.protocol, PROTOCOL_RESPONSES);
}

#[test]
fn tools_and_constraints() {
    let tools = json!([{"type": "function", "function": {
        "name": "get_weather", "description": "Weather by city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
    }}]);
    let r = render(
        PROTOCOL_CHAT_COMPLETIONS,
        json!({"model": "x", "thinking": {"type": "disabled"}, "tools": tools,
               "messages": [{"role": "user", "content": "Weather in Paris?"}]}),
    );
    assert_eq!(r.conversation.tool_choice, "auto");
    assert_eq!(r.conversation.tools[0]["function"]["name"], "get_weather");
    assert!(r.parsing_options.parse_tool_calls && !r.parsing_options.starts_in_tool_calls);
    let c = r.inference_options.output_constraint.as_ref().unwrap();
    assert_eq!(c.kind, "tool_call");
    assert!(c.schema.contains("get_weather"));

    let req = render(
        PROTOCOL_CHAT_COMPLETIONS,
        json!({"model": "x", "thinking": {"type": "disabled"}, "tools": tools, "tool_choice": "required",
               "messages": [{"role": "user", "content": "Weather?"}]}),
    );
    assert_eq!(req.conversation.tool_choice, "required");
    assert!(req.parsing_options.starts_in_tool_calls);
    let tail = CODEC
        .decode_text(&req.token_ids[req.token_ids.len() - 4..], false)
        .unwrap();
    assert!(tail.contains("DSML"), "prompt tail {tail:?}");

    let obj = render(
        PROTOCOL_CHAT_COMPLETIONS,
        json!({"model": "x", "thinking": {"type": "disabled"}, "response_format": {"type": "json_object"},
               "messages": [{"role": "user", "content": "Reply in json"}]}),
    );
    assert_eq!(obj.conversation.response_format, "json_object");
    assert!(obj.parsing_options.parse_json_output);
    assert_eq!(
        obj.inference_options
            .output_constraint
            .as_ref()
            .unwrap()
            .kind,
        "json"
    );
}

#[test]
fn context_limit_and_errors_are_protocol_shaped() {
    let mut req = request(
        PROTOCOL_CHAT_COMPLETIONS,
        json!({"model": "x", "messages": [{"role": "user", "content": "hello there"}], "max_tokens": 100}),
    );
    req.context_length = Some(20);
    let err = CODEC.render(&req).expect_err("must fail");
    assert_eq!(err.status, 400);
    assert!(
        err.body().contains("maximum context length"),
        "{}",
        err.body()
    );
    let anthropic = CODEC
        .render(&request(
            PROTOCOL_MESSAGES,
            json!({"model": "x", "messages": "nope"}),
        ))
        .unwrap_err();
    let body: Value = serde_json::from_str(&anthropic.body()).unwrap();
    assert_eq!(body["type"], "error");
    assert_eq!(body["error"]["type"], "invalid_request_error");
}

fn png_data_url(w: u32, h: u32) -> (String, Vec<u8>) {
    let img = image::RgbImage::new(w, h);
    let mut png = Vec::new();
    image::DynamicImage::ImageRgb8(img)
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .unwrap();
    let url = format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(&png)
    );
    (url, png)
}

fn user(content: Value) -> Value {
    json!({"model": "x", "thinking": {"type": "disabled"}, "messages": [{"role": "user", "content": content}]})
}

/// Every position of an item holds its pad value, and nothing else does.
fn assert_layout(r: &RenderResult) {
    let mm = r.multimodal.as_ref().unwrap();
    assert!(!r.token_ids.contains(&IMAGE_SPECIAL_TOKEN_ID));
    let mut covered = 0;
    for item in &mm.items {
        for p in &item.placements {
            assert!(p.pad_value >= mm.pad_shift);
            assert!(
                r.token_ids[p.offset..p.offset + p.length]
                    .iter()
                    .all(|t| *t == p.pad_value)
            );
            covered += p.length;
        }
        let s = item.span;
        assert!(s.offset <= item.placements[0].offset && s.offset + s.length <= r.token_ids.len());
    }
    let pads = r.token_ids.iter().filter(|t| **t >= mm.pad_shift).count();
    assert_eq!(pads, covered);
    assert_eq!(r.prompt_tokens, r.token_ids.len());
}

#[test]
fn inline_images_are_measured_and_expanded() {
    let (url, png) = png_data_url(640, 480);
    let r = render(
        PROTOCOL_CHAT_COMPLETIONS,
        user(json!([{"type": "text", "text": "What is in the image?"},
                    {"type": "image_url", "image_url": {"url": url}}])),
    );
    assert_layout(&r);
    let item = &r.multimodal.as_ref().unwrap().items[0];
    let want = ImageTokenSpec::v41().calc_token_count(640, 480).unwrap();
    assert_eq!((item.modality, item.source.kind), ("image", "base64"));
    assert_eq!(item.digest, sha256_hex(&png));
    assert_eq!(
        (item.metadata.width, item.metadata.height),
        (Some(640), Some(480))
    );
    assert_eq!(item.plan.tokens, want);
    assert_eq!(item.placements[0].length, want);
    assert_eq!(item.placements[0].pad_value, pad_value(&item.digest));
    assert_eq!(item.placements[0].reference, item.digest);
}

#[test]
fn remote_images_use_their_metadata_and_pixel_controls() {
    let digest = sha256_hex(b"remote image");
    let block = |extra: Value| {
        let mut b = json!({"type": "image_url", "image_url": {"url": "https://example.com/cat.png"},
                           "metadata": {"digest": digest, "width": 3000, "height": 2000}});
        for (k, v) in extra.as_object().unwrap() {
            b[k] = v.clone();
        }
        b
    };
    let full = render(PROTOCOL_CHAT_COMPLETIONS, user(json!([block(json!({}))])));
    assert_layout(&full);
    let spec = ImageTokenSpec::v41();
    let full_tokens = full.multimodal.as_ref().unwrap().items[0].plan.tokens;
    assert_eq!(full_tokens, spec.calc_token_count(3000, 2000).unwrap());

    let capped = render(
        PROTOCOL_CHAT_COMPLETIONS,
        user(json!([block(
            json!({"pixel_control_options": {"max_long_side": 700}})
        )])),
    );
    let (w, h) = limit_long_side(3000, 2000, 700);
    let item = &capped.multimodal.as_ref().unwrap().items[0];
    assert_eq!(
        item.plan.tokens,
        spec.calc_token_count(w as usize, h as usize).unwrap()
    );
    assert!(item.plan.tokens < full_tokens);
    assert_eq!(
        capped.prompt_tokens,
        full.prompt_tokens - full_tokens + item.plan.tokens
    );

    let mut low = block(json!({}));
    low["image_url"]["detail"] = json!("low");
    let low = render(PROTOCOL_CHAT_COMPLETIONS, user(json!([low])));
    let (w, h) = limit_long_side(3000, 2000, 512);
    assert_eq!(
        low.multimodal.as_ref().unwrap().items[0].plan.tokens,
        spec.calc_token_count(w as usize, h as usize).unwrap()
    );

    let budget = render_err(
        PROTOCOL_CHAT_COMPLETIONS,
        user(json!([block(
            json!({"pixel_control_options": {"max_total_pixels": 10000}})
        )])),
    );
    assert!(
        budget.contains("max_total_pixels 10000") && budget.contains("messages.0.content.0"),
        "{budget}"
    );
    let missing = render_err(
        PROTOCOL_CHAT_COMPLETIONS,
        user(json!([{"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}}])),
    );
    assert!(missing.contains("metadata.digest"), "{missing}");
}

#[test]
fn videos_become_timestamped_kept_frames() {
    let digest = sha256_hex(b"clip");
    let r = render(
        PROTOCOL_CHAT_COMPLETIONS,
        user(json!([
            {"type": "text", "text": "Describe the clip."},
            {"type": "video_url", "video_url": {"url": "https://example.com/clip.mp4"},
             "metadata": {"digest": digest, "width": 1280, "height": 720, "duration_ms": 3000,
                          "total_frames": 90, "video_fps": 30},
             "frame_extract_options": {"fps": 1, "pixel_control_options": {"max_long_side": 448}}}
        ])),
    );
    assert_layout(&r);
    let item = &r.multimodal.as_ref().unwrap().items[0];
    assert_eq!(item.modality, "video");
    assert_eq!(
        (item.plan.nframes, item.plan.kept_frames),
        (Some(3), Some(3))
    );
    assert_eq!(item.plan.timestamps, vec![0.0, 1.0, 2.0]);
    let (w, h) = limit_long_side(1280, 720, 448);
    let per_frame = ImageTokenSpec::v41()
        .calc_token_count(w as usize, h as usize)
        .unwrap();
    assert_eq!(item.placements.len(), 3);
    for (k, p) in item.placements.iter().enumerate() {
        assert_eq!((p.modality, p.length), ("frame", per_frame));
        assert_eq!(p.reference, kept_frame_reference(&digest, k));
        assert_eq!(p.pad_value, pad_value(&p.reference));
        assert_eq!(p.timestamp, Some(k as f64));
        // The timestamp text precedes each frame.
        let text = CODEC.encode_text(&timestamp_text(k as f64), false).unwrap();
        assert_eq!(
            &r.token_ids[p.offset - text.len()..p.offset],
            text.as_slice()
        );
    }
    let first_text = CODEC.encode_text(&timestamp_text(0.0), false).unwrap();
    assert_eq!(
        item.span.offset,
        item.placements[0].offset - first_text.len()
    );
    let last = item.placements.last().unwrap();
    assert_eq!(
        item.span.offset + item.span.length,
        last.offset + last.length
    );

    let err = render_err(
        PROTOCOL_CHAT_COMPLETIONS,
        user(
            json!([{"type": "video_url", "video_url": {"url": "https://example.com/clip.mp4"},
                     "metadata": {"digest": digest, "width": 1280, "height": 720}}]),
        ),
    );
    assert!(err.contains("duration_ms"), "{err}");
}

#[test]
fn media_blocks_of_every_protocol() {
    let digest = sha256_hex(b"img");
    let meta = json!({"digest": digest, "width": 800, "height": 600});
    let video_meta = json!({"digest": digest, "width": 640, "height": 360, "duration_ms": 2000, "video_fps": 25});
    let m = render(
        PROTOCOL_MESSAGES,
        json!({"model": "x", "max_tokens": 64, "thinking": {"type": "disabled"}, "messages": [{"role": "user", "content": [
            {"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}, "metadata": meta},
            {"type": "video", "source": {"type": "url", "url": "https://example.com/b.mp4"}, "metadata": video_meta},
            {"type": "text", "text": "Compare."}]}]}),
    );
    assert_layout(&m);
    let kinds: Vec<&str> = m
        .multimodal
        .as_ref()
        .unwrap()
        .items
        .iter()
        .map(|i| i.modality)
        .collect();
    assert_eq!(kinds, vec!["image", "video"]);
    let r = render(
        PROTOCOL_RESPONSES,
        json!({"model": "x", "reasoning": {"effort": "none"}, "input": [{"role": "user", "content": [
            {"type": "input_text", "text": "What is this?"},
            {"type": "input_image", "image_url": "https://example.com/a.png", "metadata": meta}]}]}),
    );
    assert_layout(&r);
    assert_eq!(
        r.multimodal.as_ref().unwrap().items[0].plan.tokens,
        ImageTokenSpec::v41().calc_token_count(800, 600).unwrap()
    );
}

#[test]
fn placeholder_spelling_in_user_text_stays_text() {
    let r = render(
        PROTOCOL_CHAT_COMPLETIONS,
        user(json!("<｜image｜> is just text here")),
    );
    assert!(r.multimodal.is_none());
    assert!(!r.token_ids.contains(&IMAGE_SPECIAL_TOKEN_ID));
}

#[test]
fn per_token_pieces_concatenate_to_the_text() {
    let ids = CODEC
        .encode_text("Hello </think> world <｜DSML｜ calls>", true)
        .unwrap();
    let pieces = CODEC.decode_pieces(&ids, false).unwrap();
    assert_eq!(pieces.len(), ids.len());
    assert_eq!(pieces.concat(), CODEC.decode_text(&ids, false).unwrap());
    assert!(pieces.iter().any(|p| p == "</think>"));
}

// DeepSeek V4 is text-only: media are rejected like on any text-only model,
// in every protocol, and the model reports text as its only modality.
#[test]
fn v4_is_text_only() {
    let path = format!(
        "{}/../assets/tokenizers/deepseek-v4.1/tokenizer.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let m = Model::load(&ModelConfig {
        name: "DeepSeek-V4-Pro".into(),
        family: "deepseek-v4".into(),
        tokenizer_file: Some(path.into()),
        context_length: 1_048_576,
        ..ModelConfig::default()
    })
    .expect("load");
    assert_eq!(m.info()["modalities"], json!(["text"]));
    assert_eq!(
        CODEC.info()["modalities"],
        json!(["text", "image", "video"])
    );

    let digest = sha256_hex(b"x");
    let image = json!({"type": "image_url", "image_url": {"url": "https://a/b.jpg"},
                       "metadata": {"digest": digest, "width": 640, "height": 480}});
    let video = json!({"type": "video_url", "video_url": {"url": "https://a/b.mp4"},
                       "metadata": {"digest": digest, "width": 640, "height": 480,
                                    "duration_ms": 4000, "video_fps": 10}});
    for block in [image, video] {
        let body = json!({"model": "DeepSeek-V4-Pro",
                          "messages": [{"role": "user", "content": [{"type": "text", "text": "look"}, block]}]});
        let e = m
            .render(&request("chat_completions", body))
            .expect_err("media on V4");
        assert_eq!(e.status, 400, "{}", e.body());
        assert!(
            e.body()
                .contains("model DeepSeek-V4-Pro does not accept images or videos"),
            "{}",
            e.body()
        );
    }
    let anthropic = json!({"model": "DeepSeek-V4-Pro", "max_tokens": 16, "messages": [{"role": "user", "content": [
        {"type": "image", "source": {"type": "url", "url": "https://a/b.jpg"},
         "metadata": {"digest": digest, "width": 640, "height": 480}}]}]});
    let e = m
        .render(&request("messages", anthropic))
        .expect_err("media on V4");
    assert_eq!(e.status, 400);
    assert!(e.body().contains("\"type\":\"error\""), "{}", e.body());

    // Text renders with the V4 encoding.
    let r = m
        .render(&request(
            "chat_completions",
            json!({"model": "DeepSeek-V4-Pro",
            "messages": [{"role": "user", "content": "hi"}]}),
        ))
        .expect("text on V4");
    assert!(r.multimodal.is_none() && r.prompt_tokens > 0);
    assert_eq!(r.model_version, "deepseek-v4");
}

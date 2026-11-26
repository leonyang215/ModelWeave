//! Cross-language media contract fixtures consumed by Go's mwip tests.
//! Regenerate deliberately: MW_UPDATE_MWIP_FIXTURES=1 cargo test mwip_fixtures_match

use crate::model::{Model, ModelConfig};
use crate::protocol::{PROTOCOL_CHAT_COMPLETIONS, RenderRequest};
use serde_json::{Value, json};
use std::path::PathBuf;

fn model(name: &str, family: &str, asset: &str, context_length: u32) -> Model {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../assets/tokenizers");
    Model::load(&ModelConfig {
        name: name.into(),
        family: family.into(),
        path: Some(root.join(asset)),
        tokenizer_file: (family == "qwen3-vl").then(|| root.join("qwen3/tokenizer.json")),
        context_length,
        ..ModelConfig::default()
    })
    .expect("fixture model")
}

#[test]
fn mwip_fixtures_match() {
    let tools = json!([{"type":"function","function":{"name":"get_weather","description":"Weather of a city","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer","minimum":1}},"required":["city"]}}}]);
    let image = json!({"type":"image_url","image_url":{"url":"https://img.example/a.jpg","detail":"low"},"metadata":{"digest":crate::media::sha256_hex(b"a"),"width":640,"height":480},"pixel_control_options":{"max_long_side":512}});
    let video = json!({"type":"video_url","video_url":{"url":"https://v.example/c.mp4"},"metadata":{"digest":crate::media::sha256_hex(b"c"),"width":640,"height":360,"duration_ms":2000,"total_frames":60,"video_fps":30},"frame_extract_options":{"nframes":4,"pixel_control_options":{"max_long_side":320}}});
    let media = json!([{"type":"text","text":"Describe both."},image,video]);
    let cases = [
        (
            "qwen3_tools_required.json",
            model("Qwen3-8B", "qwen3", "qwen3", 40960),
            json!({"messages":[{"role":"user","content":"Weather in Paris?"}],"tools":tools,"tool_choice":"required","chat_template_kwargs":{"enable_thinking":true}}),
        ),
        (
            "qwen3vl_media_json.json",
            model("Qwen3-VL-8B-Instruct", "qwen3-vl", "qwen3-vl", 262144),
            json!({"messages":[{"role":"user","content":media}],"max_tokens":64,"temperature":0.7,"top_p":0.8,"top_k":5,"min_p":0.05,"presence_penalty":0.5,"repetition_penalty":1.05,"response_format":{"type":"json_object"}}),
        ),
        (
            "deepseek_v41_media_tools.json",
            model(
                "DeepSeek-V4.1-Flash",
                "deepseek-v4.1",
                "deepseek-v4.1",
                1048576,
            ),
            json!({"messages":[{"role":"user","content":media}],"tools":tools,"tool_choice":"auto","thinking":{"type":"enabled"},"reasoning_effort":"high","seed":7,"logprobs":true,"top_logprobs":2,"stop":["END"]}),
        ),
    ];
    for (name, model, mut body) in cases {
        body["model"] = json!(model.meta.name);
        let result = model
            .render(&RenderRequest {
                request_id: "fixture".into(),
                protocol: PROTOCOL_CHAT_COMPLETIONS.into(),
                model: None,
                body: serde_json::value::RawValue::from_string(body.to_string()).unwrap(),
                context_length: None,
            })
            .expect("render fixture");
        let mut actual = serde_json::to_value(result).unwrap();
        actual.as_object_mut().unwrap().remove("timing_us");
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../internal/mwip/testdata")
            .join(name);
        if std::env::var_os("MW_UPDATE_MWIP_FIXTURES").is_some() {
            std::fs::write(&path, serde_json::to_string(&actual).unwrap() + "\n").unwrap();
        }
        let expected: Value =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(
            actual, expected,
            "{name}: update the intentional contract fixture with MW_UPDATE_MWIP_FIXTURES=1"
        );
    }
}

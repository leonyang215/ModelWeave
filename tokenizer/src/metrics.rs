//! Prometheus metrics of the tokenizer service.

use std::sync::LazyLock;
use std::time::Duration;

use prometheus::{
    Encoder, HistogramOpts, HistogramVec, IntCounterVec, IntGaugeVec, Opts, Registry, TextEncoder,
};

pub static REGISTRY: LazyLock<Registry> = LazyLock::new(Registry::new);

fn register<C: prometheus::core::Collector + Clone + 'static>(c: C) -> C {
    REGISTRY
        .register(Box::new(c.clone()))
        .expect("metric registration");
    c
}

/// Requests per endpoint, model, protocol and outcome.
static REQUESTS: LazyLock<IntCounterVec> = LazyLock::new(|| {
    register(
        IntCounterVec::new(
            Opts::new(
                "modelweave_tokenizer_requests_total",
                "Tokenizer requests by endpoint, model, protocol and HTTP status code.",
            ),
            &["endpoint", "model", "protocol", "code"],
        )
        .unwrap(),
    )
});

/// Render latency by stage (parse, convert, render, tokenize, media, total).
static RENDER_SECONDS: LazyLock<HistogramVec> = LazyLock::new(|| {
    register(
        HistogramVec::new(
            HistogramOpts::new(
                "modelweave_tokenizer_render_duration_seconds",
                "Time spent turning an API request into token IDs, by model and stage.",
            )
            .buckets(vec![
                0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
                1.0, 2.5,
            ]),
            &["model", "protocol", "stage"],
        )
        .unwrap(),
    )
});

/// Prompt lengths after rendering.
static PROMPT_TOKENS: LazyLock<HistogramVec> = LazyLock::new(|| {
    register(
        HistogramVec::new(
            HistogramOpts::new(
                "modelweave_tokenizer_prompt_tokens",
                "Prompt length in tokens (including multimodal expansion).",
            )
            .buckets(prometheus::exponential_buckets(64.0, 2.0, 15).unwrap()),
            &["model", "protocol"],
        )
        .unwrap(),
    )
});

static ERRORS: LazyLock<IntCounterVec> = LazyLock::new(|| {
    register(
        IntCounterVec::new(
            Opts::new(
                "modelweave_tokenizer_errors_total",
                "Render errors by model and stage (parse, convert, render, tokenize, media).",
            ),
            &["model", "stage"],
        )
        .unwrap(),
    )
});

static MEDIA: LazyLock<IntCounterVec> = LazyLock::new(|| {
    register(
        IntCounterVec::new(
            Opts::new(
                "modelweave_tokenizer_media_total",
                "Images and videos rendered into token lists, by model and modality.",
            ),
            &["model", "modality"],
        )
        .unwrap(),
    )
});

static BUILD_INFO: LazyLock<IntGaugeVec> = LazyLock::new(|| {
    register(
        IntGaugeVec::new(
            Opts::new("modelweave_tokenizer_build_info", "Build information."),
            &["version", "models"],
        )
        .unwrap(),
    )
});

static MODEL_INFO: LazyLock<IntGaugeVec> = LazyLock::new(|| {
    register(
        IntGaugeVec::new(
            Opts::new(
                "modelweave_tokenizer_model_info",
                "Models served by this tokenizer, with their family and output parsers.",
            ),
            &["model", "family", "reasoning_parser", "tool_call_parser"],
        )
        .unwrap(),
    )
});

pub fn request(endpoint: &str, model: &str, protocol: &str, code: u16) {
    REQUESTS
        .with_label_values(&[endpoint, model, protocol, &code.to_string()])
        .inc();
}

pub fn stage(model: &str, protocol: &str, stage: &str, d: Duration) {
    RENDER_SECONDS
        .with_label_values(&[model, protocol, stage])
        .observe(d.as_secs_f64());
}

pub fn prompt_tokens(model: &str, protocol: &str, n: usize) {
    PROMPT_TOKENS
        .with_label_values(&[model, protocol])
        .observe(n as f64);
}

pub fn error(model: &str, stage: &str) {
    ERRORS.with_label_values(&[model, stage]).inc();
}

pub fn media(model: &str, modality: &str) {
    MEDIA.with_label_values(&[model, modality]).inc();
}

/// One registered model.
pub struct ModelLabels<'a> {
    pub name: &'a str,
    pub family: &'a str,
    pub reasoning_parser: &'a str,
    pub tool_call_parser: &'a str,
}

/// Touch every metric so they are exported before first use, and publish
/// the build and the models.
pub fn init(models: &[ModelLabels]) {
    LazyLock::force(&REQUESTS);
    LazyLock::force(&RENDER_SECONDS);
    LazyLock::force(&PROMPT_TOKENS);
    LazyLock::force(&ERRORS);
    LazyLock::force(&MEDIA);
    let names: Vec<&str> = models.iter().map(|m| m.name).collect();
    BUILD_INFO
        .with_label_values(&[env!("CARGO_PKG_VERSION"), &names.join(",")])
        .set(1);
    for m in models {
        MODEL_INFO
            .with_label_values(&[m.name, m.family, m.reasoning_parser, m.tool_call_parser])
            .set(1);
    }
}

pub fn render_text() -> String {
    let mut buf = Vec::new();
    let encoder = TextEncoder::new();
    let families = REGISTRY.gather();
    if encoder.encode(&families, &mut buf).is_err() {
        return String::new();
    }
    String::from_utf8(buf).unwrap_or_default()
}

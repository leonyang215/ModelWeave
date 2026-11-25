//! HTTP routes of the tokenizer service.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use axum::Router;
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use serde::Deserialize;
use serde_json::json;
use tracing::{info, warn};

use crate::error::ApiError;
use crate::metrics;
use crate::model::Model;
use crate::protocol::*;

#[derive(Clone)]
pub struct AppState {
    inner: Arc<Inner>,
}

struct Inner {
    models: Vec<Arc<Model>>,
    max_body: usize,
    ready: AtomicBool,
}

impl AppState {
    pub fn new(models: Vec<Arc<Model>>, max_body: usize) -> Self {
        Self {
            inner: Arc::new(Inner {
                models,
                max_body,
                ready: AtomicBool::new(true),
            }),
        }
    }

    pub fn set_ready(&self, ready: bool) {
        self.inner.ready.store(ready, Ordering::SeqCst);
    }

    fn model(&self, model: Option<&str>, protocol: &str) -> Result<Arc<Model>, ApiError> {
        let models = &self.inner.models;
        match model {
            Some(m) if !m.is_empty() => {
                models
                    .iter()
                    .find(|c| c.matches(m))
                    .cloned()
                    .ok_or_else(|| {
                        let names: Vec<&str> =
                            models.iter().map(|c| c.meta.name.as_str()).collect();
                        ApiError::new(
                            protocol,
                            404,
                            format!(
                                "model {m:?} is not served by this tokenizer (serving {names:?})"
                            ),
                        )
                    })
            }
            _ if models.len() == 1 => Ok(models[0].clone()),
            _ => Err(ApiError::bad_request(protocol, "model is required")),
        }
    }

    fn model_for_render(&self, req: &RenderRequest) -> Result<Arc<Model>, ApiError> {
        if req.model.is_some() || self.inner.models.len() == 1 {
            return self.model(req.model.as_deref(), &req.protocol);
        }
        #[derive(Deserialize)]
        struct Peek {
            #[serde(default)]
            model: String,
        }
        let peek: Peek = serde_json::from_str(req.body.get()).map_err(|e| {
            ApiError::bad_request(&req.protocol, format!("invalid JSON request: {e}"))
        })?;
        self.model(Some(&peek.model), &req.protocol)
    }
}

pub fn router(state: AppState) -> Router {
    let max_body = state.inner.max_body;
    Router::new()
        .route("/v1/render", post(render))
        .route("/v1/tokenize_text", post(tokenize_text))
        .route("/v1/detokenize", post(detokenize))
        .route("/v1/models", get(models))
        .route("/healthz", get(|| async { "ok" }))
        .route("/readyz", get(readyz))
        .route("/metrics", get(metrics_handler))
        .layer(DefaultBodyLimit::max(max_body))
        .with_state(state)
}

fn error_response(e: &ApiError) -> Response {
    let status = StatusCode::from_u16(e.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    (
        status,
        [(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        )],
        e.body(),
    )
        .into_response()
}

fn json_response(status: StatusCode, body: String) -> Response {
    (
        status,
        [(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        )],
        body,
    )
        .into_response()
}

/// POST /v1/render: API request -> intermediate state (token list +
/// inference and parsing options).
async fn render(State(app): State<AppState>, body: Bytes) -> Response {
    let req: RenderRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
            metrics::request("render", "", "", 400);
            return error_response(&ApiError::bad_request(
                "",
                format!("invalid render request: {e}"),
            ));
        }
    };
    let protocol = req.protocol.clone();
    let request_id = req.request_id.clone();
    let mut model_name = String::new();
    let result = match app.model_for_render(&req) {
        Ok(model) => {
            model_name = model.meta.name.clone();
            let p = protocol.clone();
            tokio::task::spawn_blocking(move || model.render(&req))
                .await
                .unwrap_or_else(|e| Err(ApiError::internal(&p, format!("render task: {e}"))))
        }
        Err(e) => Err(e),
    };
    match result {
        Ok(r) => {
            let (images, videos) = r.multimodal.as_ref().map_or((0, 0), |m| {
                let videos = m.items.iter().filter(|i| i.modality == "video").count();
                (m.items.len() - videos, videos)
            });
            info!(
                request_id,
                model = model_name,
                protocol,
                prompt_tokens = r.prompt_tokens,
                max_tokens = r.inference_options.max_tokens,
                stream = r.stream,
                thinking = r.conversation.thinking_mode,
                tools = r.conversation.tools.len(),
                images,
                videos,
                total_us = r.timing_us.total,
                "rendered"
            );
            match serde_json::to_string(&r) {
                Ok(body) => {
                    metrics::request("render", &model_name, &protocol, 200);
                    json_response(StatusCode::OK, body)
                }
                Err(e) => error_response(&ApiError::internal(&protocol, e.to_string())),
            }
        }
        Err(e) => {
            warn!(request_id, model = model_name, protocol, status = e.status, error = %e.message, "render failed");
            metrics::request("render", &model_name, &protocol, e.status);
            error_response(&e)
        }
    }
}

#[derive(Deserialize)]
struct TokenizeTextRequest {
    #[serde(default)]
    model: Option<String>,
    text: String,
    #[serde(default)]
    parse_special: bool,
}

async fn tokenize_text(State(app): State<AppState>, body: Bytes) -> Response {
    let req: TokenizeTextRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return error_response(&ApiError::bad_request("", e.to_string())),
    };
    let model = match app.model(req.model.as_deref(), "") {
        Ok(c) => c,
        Err(e) => return error_response(&e),
    };
    match model.encode_text(&req.text, req.parse_special) {
        Ok(ids) => json_response(
            StatusCode::OK,
            json!({"model": model.meta.name, "token_ids": ids}).to_string(),
        ),
        Err(e) => error_response(&ApiError::internal("", e)),
    }
}

#[derive(Deserialize)]
struct DetokenizeRequest {
    #[serde(default)]
    model: Option<String>,
    token_ids: Vec<u32>,
    #[serde(default)]
    skip_special: bool,
    /// Also return the text of every token on its own.
    #[serde(default)]
    per_token: bool,
}

async fn detokenize(State(app): State<AppState>, body: Bytes) -> Response {
    let req: DetokenizeRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return error_response(&ApiError::bad_request("", e.to_string())),
    };
    let model = match app.model(req.model.as_deref(), "") {
        Ok(c) => c,
        Err(e) => return error_response(&e),
    };
    let text = match model.decode_text(&req.token_ids, req.skip_special) {
        Ok(t) => t,
        Err(e) => return error_response(&ApiError::internal("", e)),
    };
    if !req.per_token {
        return json_response(StatusCode::OK, json!({"text": text}).to_string());
    }
    match model.decode_pieces(&req.token_ids, req.skip_special) {
        Ok(pieces) => json_response(
            StatusCode::OK,
            json!({"text": text, "pieces": pieces}).to_string(),
        ),
        Err(e) => error_response(&ApiError::internal("", e)),
    }
}

async fn models(State(app): State<AppState>) -> Response {
    let data: Vec<_> = app.inner.models.iter().map(|m| m.info()).collect();
    json_response(StatusCode::OK, json!({"models": data}).to_string())
}

async fn readyz(State(app): State<AppState>) -> Response {
    if app.inner.ready.load(Ordering::SeqCst) {
        (StatusCode::OK, "ready").into_response()
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "draining").into_response()
    }
}

async fn metrics_handler() -> Response {
    (
        StatusCode::OK,
        [(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/plain; version=0.0.4"),
        )],
        metrics::render_text(),
    )
        .into_response()
}

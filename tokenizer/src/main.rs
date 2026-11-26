//! modelweave-tokenizer: converts OpenAI / Anthropic API requests into the
//! intermediate state: the conversation rendered into one token list by the
//! model's prompt format, images and videos expanded to pad-value positions,
//! with the options engines need to generate and parse. It does protocol
//! conversion and tokenization only: engines parse their output and the
//! gateway formats responses.
//!
//! One process serves every registered model (DeepSeek V4 / V4.1, Qwen3,
//! Qwen3-VL, any Hugging Face chat template); see `--config`.

mod chat;
mod deepseek;
mod error;
#[cfg(test)]
mod fixture_tests;
mod hf;
mod media;
mod metrics;
mod model;
mod protocol;
mod server;
mod template;
mod vision;

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;
use serde::Deserialize;
use tracing::{error, info};

use crate::model::{Model, ModelConfig, Renderer};

#[derive(Parser, Debug)]
#[command(name = "modelweave-tokenizer", version, about)]
struct Cli {
    /// Listen address (default: the config file's, else 0.0.0.0:8090).
    #[arg(long, env = "MW_TOKENIZER_LISTEN")]
    listen: Option<String>,
    /// TOML file with a [[models]] list; replaces the single-model flags.
    #[arg(long, env = "MW_TOKENIZER_CONFIG")]
    config: Option<PathBuf>,
    /// Canonical model name (single-model mode).
    #[arg(
        long,
        env = "MW_TOKENIZER_MODEL",
        default_value = "DeepSeek-V4.1-Flash"
    )]
    model: String,
    /// Additional accepted model names, comma separated.
    #[arg(
        long,
        env = "MW_TOKENIZER_ALIASES",
        value_delimiter = ',',
        default_value = "deepseek-v4.1,deepseek-flash"
    )]
    aliases: Vec<String>,
    /// Model family (prompt format): deepseek-v4.1, deepseek-v4, qwen3,
    /// qwen3-vl or chat-template.
    #[arg(
        long,
        visible_alias = "encoding",
        env = "MW_TOKENIZER_FAMILY",
        default_value = "deepseek-v4.1"
    )]
    family: String,
    /// Directory of the model's Hugging Face files (tokenizer.json,
    /// tokenizer_config.json, generation_config.json, ...).
    #[arg(long, env = "MW_TOKENIZER_MODEL_PATH")]
    model_path: Option<PathBuf>,
    /// Path of tokenizer.json (default: in --model-path).
    #[arg(long, env = "MW_TOKENIZER_FILE")]
    tokenizer_file: Option<PathBuf>,
    /// Context window in tokens.
    #[arg(long, env = "MW_TOKENIZER_CONTEXT_LENGTH", default_value_t = 1_048_576)]
    context_length: u32,
    /// Thinking mode when a request does not choose one.
    #[arg(long, env = "MW_TOKENIZER_DEFAULT_THINKING", default_value_t = true, action = clap::ArgAction::Set)]
    default_thinking: bool,
    /// Maximum request body size in bytes.
    #[arg(long, env = "MW_TOKENIZER_MAX_BODY_BYTES", default_value_t = 64 << 20)]
    max_body_bytes: usize,
    /// Log format: json or text.
    #[arg(long, env = "MW_LOG_FORMAT", default_value = "json")]
    log_format: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FileConfig {
    #[serde(default)]
    listen: Option<String>,
    models: Vec<ModelConfig>,
}

const DEFAULT_TOKENIZER: &str = "assets/tokenizers/deepseek-v4.1/tokenizer.json";

fn init_logging(format: &str) {
    let filter = tracing_subscriber::EnvFilter::try_from_env("MW_LOG")
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let builder = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false);
    if format == "text" {
        builder.init();
    } else {
        builder
            .json()
            .flatten_event(true)
            .with_current_span(false)
            .init();
    }
}

fn model_configs(cli: &Cli) -> Result<(Vec<ModelConfig>, Option<String>), String> {
    if let Some(path) = &cli.config {
        let text =
            std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        let fc: FileConfig =
            toml::from_str(&text).map_err(|e| format!("parse {}: {e}", path.display()))?;
        if fc.models.is_empty() {
            return Err("no models configured".into());
        }
        return Ok((fc.models, fc.listen));
    }
    let tokenizer_file = cli.tokenizer_file.clone().or_else(|| {
        cli.model_path
            .is_none()
            .then(|| PathBuf::from(DEFAULT_TOKENIZER))
    });
    let cfg = ModelConfig {
        name: cli.model.clone(),
        aliases: cli
            .aliases
            .iter()
            .filter(|a| !a.is_empty())
            .cloned()
            .collect(),
        family: cli.family.clone(),
        path: cli.model_path.clone(),
        tokenizer_file,
        context_length: cli.context_length,
        default_thinking: Some(cli.default_thinking),
        ..ModelConfig::default()
    };
    Ok((vec![cfg], None))
}

fn load_models(cli: &Cli) -> Result<(Vec<Arc<Model>>, Option<String>), String> {
    let (configs, listen) = model_configs(cli)?;
    let models = model::load_all(&configs)?;
    for m in &models {
        let info = m.info();
        info!(
            model = m.meta.name,
            aliases = ?m.aliases,
            family = m.meta.family.name(),
            context_length = m.meta.context_length,
            reasoning_parser = info["reasoning_parser"].as_str().unwrap_or("none"),
            tool_call_parser = info["tool_call_parser"].as_str().unwrap_or("none"),
            modalities = %info["modalities"],
            "model loaded"
        );
    }
    Ok((models, listen))
}

async fn shutdown_signal(state: server::AppState) {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let term = async {
        if let Ok(mut s) = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        {
            s.recv().await;
        }
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {},
        _ = term => {},
    }
    state.set_ready(false);
    info!("shutdown requested, draining");
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    init_logging(&cli.log_format);
    let (models, file_listen) = match load_models(&cli) {
        Ok(v) => v,
        Err(e) => {
            error!(error = e, "startup failed");
            std::process::exit(1);
        }
    };
    let parsers: Vec<(String, String)> = models
        .iter()
        .map(|m| {
            let (r, t) = match &m.renderer {
                Renderer::Deepseek(_) => {
                    (deepseek::PARSER.to_string(), deepseek::PARSER.to_string())
                }
                Renderer::Template(t) => (
                    t.settings.reasoning_parser.clone().unwrap_or_default(),
                    t.settings.tool_call_parser.clone().unwrap_or_default(),
                ),
            };
            (r, t)
        })
        .collect();
    let labels: Vec<metrics::ModelLabels> = models
        .iter()
        .zip(&parsers)
        .map(|(m, (r, t))| metrics::ModelLabels {
            name: &m.meta.name,
            family: m.meta.family.name(),
            reasoning_parser: r,
            tool_call_parser: t,
        })
        .collect();
    metrics::init(&labels);
    let names: Vec<String> = models.iter().map(|m| m.meta.name.clone()).collect();
    let state = server::AppState::new(models, cli.max_body_bytes);
    let app = server::router(state.clone());
    let listen = cli
        .listen
        .clone()
        .or(file_listen)
        .unwrap_or_else(|| "0.0.0.0:8090".into());
    let addr: SocketAddr = match listen.parse() {
        Ok(a) => a,
        Err(e) => {
            error!(listen, error = %e, "invalid listen address");
            std::process::exit(1);
        }
    };
    let listener = match tokio::net::TcpListener::bind(addr).await {
        Ok(l) => l,
        Err(e) => {
            error!(%addr, error = %e, "bind failed");
            std::process::exit(1);
        }
    };
    info!(%addr, models = ?names, "modelweave-tokenizer listening");
    if let Err(e) = axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal(state))
        .await
    {
        error!(error = %e, "server error");
        std::process::exit(1);
    }
}

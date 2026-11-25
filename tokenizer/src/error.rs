//! Protocol-shaped API errors.

use deepseek_recipe::request::ConversionError;
use serde_json::json;

use crate::protocol::PROTOCOL_MESSAGES;

#[derive(Debug, Clone)]
pub struct ApiError {
    pub status: u16,
    pub message: String,
    pub protocol: String,
}

impl ApiError {
    pub fn new(protocol: &str, status: u16, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
            protocol: protocol.to_string(),
        }
    }

    pub fn bad_request(protocol: &str, message: impl Into<String>) -> Self {
        Self::new(protocol, 400, message)
    }

    pub fn internal(protocol: &str, message: impl Into<String>) -> Self {
        Self::new(protocol, 500, message)
    }

    pub fn from_conversion(protocol: &str, e: ConversionError) -> Self {
        Self::new(protocol, e.status_code(), e.to_string())
    }

    /// Error body in the shape of the request's API protocol.
    pub fn body(&self) -> String {
        let v = if self.protocol == PROTOCOL_MESSAGES {
            let kind = match self.status {
                400 => "invalid_request_error",
                401 => "authentication_error",
                403 => "permission_error",
                404 => "not_found_error",
                413 => "request_too_large",
                429 => "rate_limit_error",
                503 | 529 => "overloaded_error",
                _ => "api_error",
            };
            json!({"type": "error", "error": {"type": kind, "message": self.message}})
        } else {
            let kind = if self.status < 500 {
                "invalid_request_error"
            } else {
                "internal_error"
            };
            json!({"error": {"message": self.message, "type": kind, "param": null, "code": kind}})
        };
        v.to_string()
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.message, self.status)
    }
}

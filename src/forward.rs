use axum::{
    body::Bytes,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use futures_util::StreamExt;
use serde_json::Value;
use tracing::warn;

use crate::http_error::openai_error;
use crate::scheduler::RequestCancellation;

#[derive(Clone, Debug)]
pub struct ForwardConfig {
    pub port: u16,
}

#[derive(Clone, Debug)]
pub enum ForwardError {
    UpstreamUnreachable,
    UpstreamError(u16, String),
    Cancelled,
    EmbeddingsUnsupported,
}

impl ForwardError {
    pub fn to_openai(self) -> Response {
        match self {
            Self::UpstreamUnreachable => openai_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "backend_unavailable",
                "upstream backend is not reachable".to_string(),
                None,
            ),
            Self::UpstreamError(_, message) => openai_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "upstream_error",
                message,
                None,
            ),
            Self::Cancelled => openai_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "request_cancelled",
                "request was cancelled".to_string(),
                None,
            ),
            Self::EmbeddingsUnsupported => openai_error(
                StatusCode::BAD_REQUEST,
                "unsupported_embeddings",
                "upstream backend does not support embeddings".to_string(),
                None,
            ),
        }
    }
}

impl From<reqwest::Error> for ForwardError {
    fn from(error: reqwest::Error) -> Self {
        if error.is_connect() || error.is_timeout() {
            Self::UpstreamUnreachable
        } else {
            Self::UpstreamError(
                error.status().map(|s| s.as_u16()).unwrap_or(502),
                error.to_string(),
            )
        }
    }
}

fn build_client() -> reqwest::Client {
    reqwest::Client::builder()
        .no_proxy()
        .timeout(std::time::Duration::from_secs(300))
        .build()
        .expect("reqwest client should build")
}

pub async fn forward_non_streaming(
    config: &ForwardConfig,
    path: &str,
    body: Bytes,
    cancellation: &RequestCancellation,
    model_alias: &str,
) -> Result<Response, ForwardError> {
    let client = build_client();
    let url = format!("http://127.0.0.1:{}{path}", config.port);

    let request_fut = client
        .post(&url)
        .header("content-type", "application/json")
        .body(body)
        .send();
    let result = tokio::select! {
        _ = cancellation.cancelled() => return Err(ForwardError::Cancelled),
        r = request_fut => r?,
    };

    let status = result.status();
    let response_body = result.text().await.unwrap_or_default();

    if !status.is_success() {
        if status == StatusCode::NOT_FOUND && path.starts_with("/v1/embeddings") {
            return Err(ForwardError::EmbeddingsUnsupported);
        }
        warn!(
            upstream_status = status.as_u16(),
            path = path,
            "upstream backend returned error"
        );
        return Err(ForwardError::UpstreamError(status.as_u16(), response_body));
    }

    let rewritten = rewrite_model_alias(&response_body, model_alias);

    Ok((
        StatusCode::OK,
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        rewritten,
    )
        .into_response())
}

pub async fn forward_streaming(
    config: &ForwardConfig,
    path: &str,
    body: Bytes,
    cancellation: &RequestCancellation,
    model_alias: &str,
) -> Result<Response, ForwardError> {
    let client = build_client();
    let url = format!("http://127.0.0.1:{}{path}", config.port);

    let request_fut = client
        .post(&url)
        .header("content-type", "application/json")
        .body(body)
        .send();
    let response = tokio::select! {
        _ = cancellation.cancelled() => return Err(ForwardError::Cancelled),
        r = request_fut => r?,
    };

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(ForwardError::UpstreamError(status.as_u16(), body));
    }

    let alias = model_alias.to_string();
    let stream = response.bytes_stream().map(move |chunk| {
        let alias = alias.clone();
        match chunk {
            Ok(bytes) => {
                let text = String::from_utf8_lossy(&bytes);
                let rewritten = rewrite_model_alias(&text, &alias);
                Ok::<_, reqwest::Error>(Bytes::from(rewritten.into_bytes()))
            }
            Err(e) => Err(e),
        }
    });
    let body = axum::body::Body::from_stream(stream);

    Ok((
        StatusCode::OK,
        [(axum::http::header::CONTENT_TYPE, "text/event-stream")],
        body,
    )
        .into_response())
}

fn rewrite_model_alias(body: &str, alias: &str) -> String {
    if !body.contains("\"model\"") {
        return body.to_string();
    }

    let mut parts = Vec::new();
    for part in body.split('\n') {
        let (clean_part, has_r) = if let Some(stripped) = part.strip_suffix('\r') {
            (stripped, true)
        } else {
            (part, false)
        };

        let processed = if let Some(sse_data) = clean_part.strip_prefix("data: ") {
            if sse_data.contains("\"model\"") {
                match serde_json::from_str::<Value>(sse_data) {
                    Ok(mut value) => {
                        if let Some(obj) = value.as_object_mut()
                            && let Some(model) = obj.get_mut("model")
                        {
                            *model = Value::String(alias.to_string());
                        }
                        if let Ok(rewritten) = serde_json::to_string(&value) {
                            format!("data: {rewritten}")
                        } else {
                            clean_part.to_string()
                        }
                    }
                    Err(_) => clean_part.to_string(),
                }
            } else {
                clean_part.to_string()
            }
        } else if clean_part.contains("\"model\"") {
            match serde_json::from_str::<Value>(clean_part) {
                Ok(mut value) => {
                    if let Some(obj) = value.as_object_mut()
                        && let Some(model) = obj.get_mut("model")
                    {
                        *model = Value::String(alias.to_string());
                    }
                    if let Ok(rewritten) = serde_json::to_string(&value) {
                        rewritten
                    } else {
                        clean_part.to_string()
                    }
                }
                Err(_) => clean_part.to_string(),
            }
        } else {
            clean_part.to_string()
        };

        if has_r {
            parts.push(processed + "\r");
        } else {
            parts.push(processed);
        }
    }

    parts.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rewrite_model_alias_non_streaming() {
        let json_body = r#"{"model":"llama-3-8b","choices":[]}"#;
        let rewritten = rewrite_model_alias(json_body, "my-custom-model");
        let parsed: Value = serde_json::from_str(&rewritten).unwrap();
        assert_eq!(parsed["model"], "my-custom-model");
    }

    #[test]
    fn test_rewrite_model_alias_streaming() {
        let sse_chunk = r#"data: {"model":"llama-3-8b","choices":[]}"#;
        let rewritten = rewrite_model_alias(sse_chunk, "my-custom-model");
        assert!(
            rewritten.contains("my-custom-model"),
            "Expected model to be rewritten, got: {}",
            rewritten
        );
    }
}

use axum::{
    body::Bytes,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde_json::{Value, json};

use crate::scheduler::RequestCancellation;

pub struct ForwardConfig {
    pub port: u16,
}

#[derive(Debug)]
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
                "upstream backend is not reachable",
                None,
            ),
            Self::UpstreamError(_, message) => openai_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "upstream_error",
                &message,
                None,
            ),
            Self::Cancelled => openai_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "request_cancelled",
                "request was cancelled",
                None,
            ),
            Self::EmbeddingsUnsupported => openai_error(
                StatusCode::BAD_REQUEST,
                "unsupported_embeddings",
                "upstream backend does not support embeddings",
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

    let stream = response.bytes_stream();
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

    match serde_json::from_str::<Value>(body) {
        Ok(mut value) => {
            if let Some(obj) = value.as_object_mut() {
                if let Some(model) = obj.get_mut("model") {
                    *model = Value::String(alias.to_string());
                }
            }
            serde_json::to_string(&value).unwrap_or_else(|_| body.to_string())
        }
        Err(_) => body.to_string(),
    }
}

fn openai_error(status: StatusCode, code: &str, message: &str, retry_after: Option<&str>) -> Response {
    let mut response = (
        status,
        axum::Json(json!({
            "error": {
                "message": message,
                "type": code,
                "param": Value::Null,
                "code": code,
            }
        })),
    )
        .into_response();

    if let Some(retry_after) = retry_after {
        response.headers_mut().insert(
            axum::http::header::RETRY_AFTER,
            retry_after.parse().expect("valid retry-after header"),
        );
    }

    response
}
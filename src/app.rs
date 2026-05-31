use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::state::{AppState, LifecycleState};

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/status", get(status))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/embeddings", post(embeddings))
        .with_state(state)
}

async fn healthz() -> impl IntoResponse {
    Json(json!({ "status": "ok" }))
}

async fn status(State(state): State<AppState>) -> impl IntoResponse {
    let backend = state.backend.read().await;

    Json(json!({
        "state": backend.lifecycle.as_str(),
        "model_alias": state.config.model.alias,
        "capabilities": {
            "chat": backend.chat.as_str(),
            "embeddings": backend.embeddings.as_str(),
            "embeddings_reason": backend.embeddings_reason,
        },
        "last_wake_attempt_at": backend.last_wake_attempt_at,
        "lease_expires_at": backend.lease_expires_at,
        "tunnel": {
            "state": backend.tunnel.as_str(),
        },
        "units": {
            "llama_server": backend.llama_server_unit.as_str(),
            "inhibit_holder": backend.inhibit_unit.as_str(),
        }
    }))
}

async fn list_models(State(state): State<AppState>) -> impl IntoResponse {
    Json(json!({
        "object": "list",
        "data": [
            {
                "id": state.config.model.alias,
                "object": "model",
                "created": 0,
                "owned_by": state.config.model.owned_by,
                "provider": state.config.model.provider_id,
            }
        ]
    }))
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(payload): Json<ChatCompletionRequest>,
) -> Response {
    if payload.messages.is_empty() {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "messages must not be empty".to_string(),
            None,
        );
    }

    if let Some(role) = payload.invalid_role() {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "unsupported_role",
            format!("unsupported message role '{role}'"),
            None,
        );
    }

    if payload.has_null_content() {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "message content must not be null".to_string(),
            None,
        );
    }

    if payload.model != state.config.model.alias {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "model_not_found",
            format!(
                "unsupported model '{}': expected '{}'",
                payload.model, state.config.model.alias
            ),
            None,
        );
    }

    if payload.stream {
        return openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "warming_up",
            "streaming is not available until backend orchestration is implemented".to_string(),
            Some("10"),
        );
    }

    let mut backend = state.backend.write().await;
    if matches!(backend.lifecycle, LifecycleState::Cold) {
        backend.lifecycle = LifecycleState::Warming;
    }

    openai_error(
        StatusCode::SERVICE_UNAVAILABLE,
        "warming_up",
        "backend wake and forwarding are not implemented yet".to_string(),
        Some("10"),
    )
}

async fn embeddings(
    State(state): State<AppState>,
    Json(payload): Json<EmbeddingsRequest>,
) -> Response {
    if payload.input.is_null() {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "input must not be null".to_string(),
            None,
        );
    }

    if payload.model != state.config.model.alias {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "model_not_found",
            format!(
                "unsupported model '{}': expected '{}'",
                payload.model, state.config.model.alias
            ),
            None,
        );
    }

    if !state.config.embeddings.enabled {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "unsupported_embeddings",
            "embeddings are disabled by configuration".to_string(),
            None,
        );
    }

    let mut backend = state.backend.write().await;
    if matches!(backend.lifecycle, LifecycleState::Cold) {
        backend.lifecycle = LifecycleState::Warming;
    }

    openai_error(
        StatusCode::SERVICE_UNAVAILABLE,
        "warming_up",
        "backend wake and forwarding are not implemented yet".to_string(),
        Some("10"),
    )
}

fn openai_error(
    status: StatusCode,
    code: &str,
    message: String,
    retry_after: Option<&str>,
) -> Response {
    let mut response = (
        status,
        Json(json!({
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

#[derive(Debug, Deserialize)]
struct ChatCompletionRequest {
    model: String,
    #[serde(default)]
    stream: bool,
    messages: Vec<ChatMessage>,
}

impl ChatCompletionRequest {
    fn invalid_role(&self) -> Option<&str> {
        self.messages
            .iter()
            .find(|message| {
                !matches!(
                    message.role.as_str(),
                    "system" | "user" | "assistant" | "tool" | "developer"
                )
            })
            .map(|message| message.role.as_str())
    }

    fn has_null_content(&self) -> bool {
        self.messages
            .iter()
            .any(|message| message.content.is_null())
    }
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    role: String,
    content: Value,
}

#[derive(Debug, Deserialize)]
struct EmbeddingsRequest {
    model: String,
    input: Value,
}

#[cfg(test)]
mod tests {
    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use http_body_util::BodyExt;
    use tower::util::ServiceExt;

    use super::*;
    use crate::{
        config::{AppConfig, EmbeddingsConfig, ModelConfig},
        state::AppState,
    };

    fn test_state() -> AppState {
        AppState::new(AppConfig {
            listen_port: 3000,
            model: ModelConfig {
                alias: "proxy-model".to_string(),
                provider_id: "llama.cpp".to_string(),
                owned_by: "test-suite".to_string(),
            },
            embeddings: EmbeddingsConfig { enabled: true },
        })
    }

    #[tokio::test]
    async fn healthz_returns_ok() {
        let app = build_router(test_state());
        let response = app
            .oneshot(Request::get("/healthz").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(body, r#"{"status":"ok"}"#);
    }

    #[tokio::test]
    async fn models_returns_configured_alias_without_waking_backend() {
        let state = test_state();
        let app = build_router(state.clone());

        let response = app
            .oneshot(Request::get("/v1/models").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let backend = state.backend.read().await;
        assert!(matches!(backend.lifecycle, LifecycleState::Cold));
    }

    #[tokio::test]
    async fn chat_unknown_model_returns_bad_request() {
        let app = build_router(test_state());
        let response = app
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"wrong","messages":[{"role":"user","content":"hi"}]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn chat_marks_backend_warming_and_returns_retryable_503() {
        let state = test_state();
        let app = build_router(state.clone());

        let response = app
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"proxy-model","messages":[{"role":"user","content":"hi"}]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(response.headers()["retry-after"], "10");

        let backend = state.backend.read().await;
        assert!(matches!(backend.lifecycle, LifecycleState::Warming));
    }

    #[tokio::test]
    async fn chat_rejects_empty_messages() {
        let app = build_router(test_state());
        let response = app
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"model":"proxy-model","messages":[]}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn embeddings_reject_null_input() {
        let app = build_router(test_state());
        let response = app
            .oneshot(
                Request::post("/v1/embeddings")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"model":"proxy-model","input":null}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn embeddings_can_be_disabled_independently() {
        let state = AppState::new(AppConfig {
            listen_port: 3000,
            model: ModelConfig {
                alias: "proxy-model".to_string(),
                provider_id: "llama.cpp".to_string(),
                owned_by: "test-suite".to_string(),
            },
            embeddings: EmbeddingsConfig { enabled: false },
        });
        let app = build_router(state);

        let response = app
            .oneshot(
                Request::post("/v1/embeddings")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"model":"proxy-model","input":"hello"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn status_exposes_backend_shape() {
        let app = build_router(test_state());
        let response = app
            .oneshot(Request::get("/status").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json["state"], "cold");
        assert_eq!(json["model_alias"], "proxy-model");
        assert_eq!(json["capabilities"]["chat"], "ready");
        assert_eq!(json["capabilities"]["embeddings"], "ready");
    }
}

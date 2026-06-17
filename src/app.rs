use axum::extract::FromRequestParts;
use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    http::request::Parts,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    http_error::openai_error,
    lifecycle::{BackendStatus, CapabilityState, LifecycleRequest},
    orchestrate::Orchestrator,
    state::AppState,
};

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/status", get(status))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/embeddings", post(embeddings))
        .with_state(state)
}

struct OptionalAuthorization;

impl<S> FromRequestParts<S> for OptionalAuthorization
where
    S: Send + Sync,
{
    type Rejection = Response;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        let _ = parts.headers.get(axum::http::header::AUTHORIZATION);
        Ok(Self)
    }
}

struct RequireJsonContentType;

impl<S> FromRequestParts<S> for RequireJsonContentType
where
    S: Send + Sync,
{
    type Rejection = Response;

    async fn from_request_parts(parts: &mut Parts, _state: &S) -> Result<Self, Self::Rejection> {
        let has_json = parts
            .headers
            .get(axum::http::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|ct| ct.contains("application/json"));

        if has_json {
            Ok(Self)
        } else {
            Err(openai_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "content-type must be application/json".to_string(),
                None,
            ))
        }
    }
}

fn normalize_json_error(err: serde_json::Error) -> String {
    let msg = err.to_string();

    if let Some(field) = extract_quoted_value(&msg, "unknown field `") {
        return format!("unsupported field '{field}'");
    }

    if msg.contains("unknown variant") {
        return "request body contains invalid fields".to_string();
    }

    if msg.contains("missing field `tool_call_id`") {
        return "tool messages must include tool_call_id".to_string();
    }

    if msg.contains("missing field `content`") {
        return "messages must include content unless assistant tool_calls are present".to_string();
    }

    if msg.contains("invalid type") {
        return "request body contains an invalid field type".to_string();
    }

    if msg.contains("data did not match any variant of untagged enum") {
        return "request body contains invalid fields".to_string();
    }

    "request body contains invalid fields".to_string()
}

fn extract_quoted_value<'a>(message: &'a str, prefix: &str) -> Option<&'a str> {
    let suffix = &message[message.find(prefix)? + prefix.len()..];
    let end = suffix.find('`')?;
    Some(&suffix[..end])
}

async fn healthz() -> impl IntoResponse {
    Json(json!({ "status": "ok" }))
}

async fn status(State(state): State<AppState>) -> impl IntoResponse {
    let metrics = state.metrics.snapshot();

    let chat_status = state.chat_lifecycle.status();
    let chat = backend_status_json(
        &chat_status,
        &state.config.model.alias,
        chat_status.chat,
        None,
    );

    let embeddings = if state.config.embeddings.enabled {
        let embeddings_status = state.embeddings_lifecycle.status();
        let capability = embeddings_status.embeddings;
        let capability_reason = embeddings_status.embeddings_reason.clone();
        Some(backend_status_json(
            &embeddings_status,
            state.model_alias_for(&LifecycleRequest::Embeddings),
            capability,
            capability_reason,
        ))
    } else {
        None
    };

    Json(json!({
        "chat": chat,
        "embeddings": embeddings,
        "metrics": metrics,
    }))
}

fn backend_status_json(
    status: &BackendStatus,
    model_alias: &str,
    capability: CapabilityState,
    capability_reason: Option<String>,
) -> Value {
    json!({
        "model_alias": model_alias,
        "state": status.lifecycle,
        "capability": capability,
        "capability_reason": capability_reason,
        "tunnel": status.tunnel,
        "last_wake_attempt_at": status.last_wake_attempt_at,
        "lease_expires_at": status.lease_expires_at,
        "host_unit": {
            "llama_server_unit": status.llama_server_unit,
            "inhibit_unit": status.inhibit_unit,
        },
    })
}

async fn list_models(State(state): State<AppState>) -> impl IntoResponse {
    let mut data = vec![json!({
        "id": state.config.model.alias,
        "object": "model",
        "created": 0,
        "owned_by": state.config.model.owned_by,
        "provider": state.config.model.provider_id,
    })];

    if let Some(backend) = &state.config.embeddings.backend {
        data.push(json!({
            "id": backend.alias,
            "object": "model",
            "created": 0,
            "owned_by": backend.owned_by,
            "provider": backend.provider_id,
        }));
    }

    Json(json!({
        "object": "list",
        "data": data,
    }))
}

async fn chat_completions(
    State(state): State<AppState>,
    _: RequireJsonContentType,
    _: OptionalAuthorization,
    raw_body: axum::body::Bytes,
) -> Response {
    state.metrics.inc_chat_requests();

    let payload: ChatCompletionRequest = match serde_json::from_slice(&raw_body) {
        Ok(v) => v,
        Err(e) => {
            return openai_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                normalize_json_error(e),
                None,
            );
        }
    };

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

    if let Some(message) = payload.content_error() {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            message.to_string(),
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

    Orchestrator::new(state)
        .execute(LifecycleRequest::Chat, raw_body, payload.stream)
        .await
}

async fn embeddings(
    State(state): State<AppState>,
    _: RequireJsonContentType,
    _: OptionalAuthorization,
    raw_body: axum::body::Bytes,
) -> Response {
    state.metrics.inc_embeddings_requests();
    let payload: EmbeddingsRequest = match serde_json::from_slice(&raw_body) {
        Ok(v) => v,
        Err(e) => {
            return openai_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                normalize_json_error(e),
                None,
            );
        }
    };

    if payload.input.is_null() {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "input must not be null".to_string(),
            None,
        );
    }

    let embeddings_alias = state.model_alias_for(&LifecycleRequest::Embeddings);
    if payload.model != embeddings_alias {
        return openai_error(
            StatusCode::BAD_REQUEST,
            "model_not_found",
            format!(
                "unsupported model '{}': expected '{}'",
                payload.model, embeddings_alias
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

    Orchestrator::new(state)
        .execute(LifecycleRequest::Embeddings, raw_body, false)
        .await
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatCompletionRequest {
    model: String,
    #[serde(default)]
    stream: bool,
    messages: Vec<ChatMessage>,
    #[serde(rename = "tools")]
    _tools: Option<Vec<ChatTool>>,
    #[serde(rename = "tool_choice")]
    _tool_choice: Option<ToolChoice>,
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

    fn content_error(&self) -> Option<&'static str> {
        self.messages.iter().find_map(ChatMessage::content_error)
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatMessage {
    role: String,
    content: Option<Value>,
    tool_calls: Option<Vec<ToolCall>>,
    #[serde(rename = "tool_call_id")]
    tool_call_id: Option<String>,
}

impl ChatMessage {
    fn content_error(&self) -> Option<&'static str> {
        match self.role.as_str() {
            "assistant" => {
                if self.tool_call_id.is_some() {
                    return Some("only tool messages may include tool_call_id");
                }

                if self.tool_calls.as_ref().is_some_and(Vec::is_empty) {
                    return Some("assistant messages tool_calls must not be empty");
                }

                if self.content.as_ref().is_none_or(Value::is_null) && self.tool_calls.is_none() {
                    Some("message content must not be null")
                } else {
                    None
                }
            }
            "tool" => {
                if self.tool_calls.is_some() {
                    return Some("only assistant messages may include tool_calls");
                }

                if self.tool_call_id.is_none() {
                    Some("tool messages must include tool_call_id")
                } else if self.content.as_ref().is_none_or(Value::is_null) {
                    Some("message content must not be null")
                } else {
                    None
                }
            }
            _ => {
                if self.tool_calls.is_some() {
                    return Some("only assistant messages may include tool_calls");
                }

                if self.tool_call_id.is_some() {
                    return Some("only tool messages may include tool_call_id");
                }

                self.content
                    .as_ref()
                    .is_none_or(Value::is_null)
                    .then_some("message content must not be null")
            }
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatTool {
    #[serde(rename = "type")]
    _kind: ToolKind,
    #[serde(rename = "function")]
    _function: ToolFunction,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ToolFunction {
    #[serde(rename = "name")]
    _name: String,
    #[serde(default)]
    #[serde(rename = "description")]
    _description: Option<String>,
    #[serde(default)]
    #[serde(rename = "parameters")]
    _parameters: Option<Value>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ToolChoice {
    _Mode(ToolChoiceMode),
    _Named(NamedToolChoice),
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
enum ToolChoiceMode {
    #[serde(rename = "none")]
    None,
    #[serde(rename = "auto")]
    Auto,
    #[serde(rename = "required")]
    Required,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct NamedToolChoice {
    #[serde(rename = "type")]
    _kind: ToolKind,
    #[serde(rename = "function")]
    _function: ToolChoiceFunction,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ToolChoiceFunction {
    #[serde(rename = "name")]
    _name: String,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ToolCall {
    #[serde(rename = "id")]
    _id: String,
    #[serde(rename = "type")]
    _kind: ToolKind,
    #[serde(rename = "function")]
    _function: ToolCallFunction,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
enum ToolKind {
    #[serde(rename = "function")]
    Function,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ToolCallFunction {
    #[serde(rename = "name")]
    _name: String,
    #[serde(rename = "arguments")]
    _arguments: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct EmbeddingsRequest {
    model: String,
    input: Value,
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use http_body_util::BodyExt;
    use tower::util::ServiceExt;

    use super::*;
    use crate::{
        config::{AppConfig, EmbeddingsConfig, ModelConfig, WarmExecutionConfig},
        lifecycle::{
            BackendStatus, CapabilityState, LifecycleDecision, LifecycleError,
            LifecycleOrchestrator, LifecycleState, Timestamp, TunnelState, UnitState,
        },
        scheduler::WarmExecutionScheduler,
        state::AppState,
    };

    #[derive(Clone)]
    struct StaticLifecycleOrchestrator {
        decision: LifecycleDecision,
    }

    impl LifecycleOrchestrator for StaticLifecycleOrchestrator {
        fn ensure_backend(
            &self,
            _request: LifecycleRequest,
        ) -> crate::lifecycle::LifecycleFuture<'_, LifecycleDecision> {
            let decision = self.decision.clone();
            Box::pin(async move { decision })
        }

        fn status(&self) -> BackendStatus {
            match &self.decision {
                LifecycleDecision::Ready(status) => status.clone(),
                LifecycleDecision::Warming { status, .. } => status.clone(),
                LifecycleDecision::Failed { status, .. } => status.clone(),
            }
        }

        fn degrade_embeddings(&self, _reason: String) -> crate::lifecycle::LifecycleFuture<'_, ()> {
            Box::pin(async move {})
        }

        fn mark_warming(&self) {}
    }

    fn test_config() -> AppConfig {
        use crate::config::HostConfig;

        AppConfig {
            listen_port: 3000,
            model: ModelConfig {
                alias: "proxy-model".to_string(),
                provider_id: "llama.cpp".to_string(),
                owned_by: "test-suite".to_string(),
            },
            embeddings: EmbeddingsConfig {
                enabled: true,
                backend: None,
            },
            warm_execution: WarmExecutionConfig::default(),
            host: HostConfig::default(),
            cold_start_max_waiting: 32,
        }
    }

    fn test_state() -> AppState {
        AppState::new(test_config())
    }

    fn state_with_lifecycle(decision: LifecycleDecision) -> AppState {
        AppState::with_lifecycle(
            test_config(),
            Arc::new(StaticLifecycleOrchestrator { decision }),
        )
    }

    fn dual_mode_config() -> AppConfig {
        use crate::config::EmbeddingsBackendConfig;

        AppConfig {
            embeddings: EmbeddingsConfig {
                enabled: true,
                backend: Some(EmbeddingsBackendConfig {
                    alias: "embed-model".to_string(),
                    provider_id: "llama.cpp".to_string(),
                    owned_by: "test-suite".to_string(),
                    model_path: "/models/embed.gguf".to_string(),
                    tunnel_local_port: 18081,
                    remote_port: 8081,
                }),
            },
            ..test_config()
        }
    }

    fn dual_mode_state(chat: LifecycleDecision, embeddings: LifecycleDecision) -> AppState {
        AppState::with_dual_lifecycles(
            dual_mode_config(),
            Arc::new(StaticLifecycleOrchestrator { decision: chat }),
            Arc::new(StaticLifecycleOrchestrator {
                decision: embeddings,
            }),
        )
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
        let status = state.chat_lifecycle.status();
        assert!(matches!(status.lifecycle, LifecycleState::Cold));
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
    async fn chat_rejects_unknown_top_level_field() {
        let app = build_router(test_state());
        let response = app
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"proxy-model","messages":[{"role":"user","content":"hi"}],"temperature":0.2}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["type"], "invalid_request_error");
        assert_eq!(json["error"]["message"], "unsupported field 'temperature'");
    }

    #[tokio::test]
    async fn chat_rejects_unknown_message_field() {
        let app = build_router(test_state());
        let response = app
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"proxy-model","messages":[{"role":"user","content":"hi","name":"extra"}]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["message"], "unsupported field 'name'");
    }

    #[tokio::test]
    async fn chat_warming_decision_maps_to_retryable_503() {
        let warming_status = BackendStatus {
            lifecycle: LifecycleState::Warming,
            tunnel: TunnelState::Down,
            ..BackendStatus::default()
        };
        let app = build_router(state_with_lifecycle(LifecycleDecision::Warming {
            status: warming_status,
            retry_after_secs: 10,
        }));

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
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["type"], "warming_up");
    }

    #[tokio::test]
    async fn chat_rejects_null_assistant_content_without_tool_calls() {
        let app = build_router(test_state());
        let response = app
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"proxy-model","messages":[{"role":"assistant","content":null}]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn chat_accepts_tool_calling_fields_and_null_assistant_content() {
        let warming_status = BackendStatus {
            lifecycle: LifecycleState::Warming,
            tunnel: TunnelState::Down,
            ..BackendStatus::default()
        };
        let app = build_router(state_with_lifecycle(LifecycleDecision::Warming {
            status: warming_status,
            retry_after_secs: 10,
        }));

        let response = app
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"proxy-model","messages":[{"role":"user","content":"hi"},{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}"}}]},{"role":"tool","tool_call_id":"call_1","content":"{}"}],"tools":[{"type":"function","function":{"name":"lookup","description":"Lookup data","parameters":{"type":"object"}}}],"tool_choice":"auto"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(response.headers()["retry-after"], "10");
    }

    #[tokio::test]
    async fn chat_accepts_authorization_header_identically() {
        let warming_status = BackendStatus {
            lifecycle: LifecycleState::Warming,
            tunnel: TunnelState::Down,
            ..BackendStatus::default()
        };
        let app = build_router(state_with_lifecycle(LifecycleDecision::Warming {
            status: warming_status,
            retry_after_secs: 10,
        }));

        let response = app
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer test-token")
                    .body(Body::from(
                        r#"{"model":"proxy-model","messages":[{"role":"user","content":"hi"}]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(response.headers()["retry-after"], "10");
    }

    #[tokio::test]
    async fn chat_rejects_tool_message_without_tool_call_id() {
        let app = build_router(test_state());
        let response = app
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"proxy-model","messages":[{"role":"tool","content":"{}"}]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            json["error"]["message"],
            "tool messages must include tool_call_id"
        );
    }

    #[tokio::test]
    async fn chat_rejects_invalid_tool_calls_shape() {
        let app = build_router(test_state());
        let response = app
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"proxy-model","messages":[{"role":"assistant","content":null,"tool_calls":"bad"}]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            json["error"]["message"],
            "request body contains an invalid field type"
        );
    }

    #[tokio::test]
    async fn chat_rejects_invalid_tool_choice_shape() {
        let app = build_router(test_state());
        let response = app
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"proxy-model","messages":[{"role":"user","content":"hi"}],"tool_choice":123}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            json["error"]["message"],
            "request body contains invalid fields"
        );
    }

    #[tokio::test]
    async fn chat_rejects_invalid_tool_choice_value() {
        let app = build_router(test_state());
        let response = app
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"proxy-model","messages":[{"role":"user","content":"hi"}],"tool_choice":"banana"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            json["error"]["message"],
            "request body contains invalid fields"
        );
    }

    #[tokio::test]
    async fn chat_rejects_invalid_tool_type() {
        let app = build_router(test_state());
        let response = app
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"proxy-model","messages":[{"role":"user","content":"hi"}],"tools":[{"type":"custom","function":{"name":"lookup"}}]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            json["error"]["message"],
            "request body contains invalid fields"
        );
    }

    #[tokio::test]
    async fn chat_rejects_empty_assistant_tool_calls() {
        let app = build_router(test_state());
        let response = app
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"proxy-model","messages":[{"role":"assistant","content":null,"tool_calls":[]}]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            json["error"]["message"],
            "assistant messages tool_calls must not be empty"
        );
    }

    #[tokio::test]
    async fn chat_rejects_tool_calls_on_user_message() {
        let app = build_router(test_state());
        let response = app
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"proxy-model","messages":[{"role":"user","content":"hi","tool_calls":[]}]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            json["error"]["message"],
            "only assistant messages may include tool_calls"
        );
    }

    #[tokio::test]
    async fn chat_rejects_tool_call_id_on_assistant_message() {
        let app = build_router(test_state());
        let response = app
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"proxy-model","messages":[{"role":"assistant","content":"hi","tool_call_id":"call_1"}]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            json["error"]["message"],
            "only tool messages may include tool_call_id"
        );
    }

    #[tokio::test]
    async fn chat_rejects_missing_json_content_type_with_stable_message() {
        let app = build_router(test_state());
        let response = app
            .oneshot(
                Request::post("/v1/chat/completions")
                    .body(Body::from(
                        r#"{"model":"proxy-model","messages":[{"role":"user","content":"hi"}]}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            json["error"]["message"],
            "content-type must be application/json"
        );
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
        use crate::config::HostConfig;

        let state = AppState::new(AppConfig {
            listen_port: 3000,
            model: ModelConfig {
                alias: "proxy-model".to_string(),
                provider_id: "llama.cpp".to_string(),
                owned_by: "test-suite".to_string(),
            },
            embeddings: EmbeddingsConfig {
                enabled: false,
                backend: None,
            },
            warm_execution: WarmExecutionConfig::default(),
            host: HostConfig::default(),
            cold_start_max_waiting: 32,
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
    async fn embeddings_reject_unknown_field() {
        let app = build_router(test_state());
        let response = app
            .oneshot(
                Request::post("/v1/embeddings")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"proxy-model","input":"hello","encoding_format":"float"}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            json["error"]["message"],
            "unsupported field 'encoding_format'"
        );
    }

    #[tokio::test]
    async fn embeddings_accepts_authorization_header_identically() {
        let warming_status = BackendStatus {
            lifecycle: LifecycleState::Warming,
            tunnel: TunnelState::Down,
            ..BackendStatus::default()
        };
        let app = build_router(state_with_lifecycle(LifecycleDecision::Warming {
            status: warming_status,
            retry_after_secs: 10,
        }));

        let response = app
            .oneshot(
                Request::post("/v1/embeddings")
                    .header("content-type", "application/json")
                    .header("authorization", "Bearer test-token")
                    .body(Body::from(r#"{"model":"proxy-model","input":"hello"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(response.headers()["retry-after"], "10");
    }

    #[tokio::test]
    async fn chat_ready_decision_maps_to_backend_unavailable_without_retry_after() {
        let ready_status = BackendStatus {
            lifecycle: LifecycleState::Ready,
            tunnel: TunnelState::Ready,
            ..BackendStatus::default()
        };
        let app = build_router(state_with_lifecycle(LifecycleDecision::Ready(ready_status)));

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
        assert!(response.headers().get("retry-after").is_none());
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["type"], "backend_unavailable");
    }

    #[tokio::test]
    async fn embeddings_failed_decision_maps_to_backend_error() {
        let failed_status = BackendStatus {
            lifecycle: LifecycleState::Error,
            ..BackendStatus::default()
        };
        let app = build_router(state_with_lifecycle(LifecycleDecision::Failed {
            status: failed_status,
            error: LifecycleError::new("helper command failed"),
        }));

        let response = app
            .oneshot(
                Request::post("/v1/embeddings")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"model":"proxy-model","input":"hello"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(response.headers().get("retry-after").is_none());
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["type"], "backend_error");
        assert_eq!(json["error"]["message"], "helper command failed");
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

        assert_eq!(json["chat"]["state"], "cold");
        assert_eq!(json["chat"]["model_alias"], "proxy-model");
        assert_eq!(json["chat"]["capability"], "ready");
        assert_eq!(json["embeddings"]["capability"], "ready");
    }

    #[tokio::test]
    async fn status_serializes_timestamps_tunnel_and_units() {
        let backend = BackendStatus {
            lifecycle: LifecycleState::Ready,
            chat: CapabilityState::Degraded,
            embeddings: CapabilityState::Ready,
            embeddings_reason: Some("remote disabled".to_string()),
            tunnel: TunnelState::Ready,
            last_wake_attempt_at: Some(Timestamp::new(1_717_156_800)),
            lease_expires_at: Some(Timestamp::new(1_717_158_600)),
            llama_server_unit: UnitState::Active,
            inhibit_unit: UnitState::Activating,
        };
        let app = build_router(state_with_lifecycle(LifecycleDecision::Warming {
            status: backend,
            retry_after_secs: 10,
        }));

        let response = app
            .oneshot(Request::get("/status").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json["chat"]["state"], "ready");
        assert_eq!(json["chat"]["capability"], "degraded");
        assert_eq!(json["embeddings"]["capability_reason"], "remote disabled");
        assert_eq!(json["chat"]["tunnel"], "ready");
        assert_eq!(json["chat"]["host_unit"]["llama_server_unit"], "active");
        assert_eq!(json["chat"]["host_unit"]["inhibit_unit"], "activating");
        assert_eq!(json["chat"]["last_wake_attempt_at"], 1717156800u64);
        assert_eq!(json["chat"]["lease_expires_at"], 1717158600u64);
    }

    #[tokio::test]
    async fn status_reflects_orchestrator_snapshot() {
        let app = build_router(state_with_lifecycle(LifecycleDecision::Warming {
            status: BackendStatus {
                lifecycle: LifecycleState::Warming,
                tunnel: TunnelState::Connecting,
                ..BackendStatus::default()
            },
            retry_after_secs: 10,
        }));

        let response = app
            .oneshot(Request::get("/status").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["chat"]["state"], "warming");
        assert_eq!(json["chat"]["tunnel"], "connecting");
    }

    #[tokio::test]
    async fn ready_embeddings_request_times_out_in_shared_warm_queue_with_429() {
        let ready_status = BackendStatus {
            lifecycle: LifecycleState::Ready,
            tunnel: TunnelState::Ready,
            ..BackendStatus::default()
        };
        let config = AppConfig {
            warm_execution: WarmExecutionConfig {
                max_active_requests: 1,
                max_queued_requests: 1,
                queue_timeout: std::time::Duration::from_millis(20),
            },
            ..test_config()
        };
        let scheduler = WarmExecutionScheduler::new(config.warm_execution.clone());
        let (release_tx, release_rx) = tokio::sync::oneshot::channel::<()>();
        let held_slot = tokio::spawn({
            let scheduler = scheduler.clone();
            async move {
                scheduler
                    .execute(LifecycleRequest::Chat, |_| async move {
                        let _ = release_rx.await;
                    })
                    .await
                    .unwrap();
            }
        });
        tokio::task::yield_now().await;

        let app = build_router(AppState::with_services(
            config,
            Arc::new(StaticLifecycleOrchestrator {
                decision: LifecycleDecision::Ready(ready_status),
            }),
            scheduler,
        ));

        let response = app
            .oneshot(
                Request::post("/v1/embeddings")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"model":"proxy-model","input":"hello"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["type"], "overloaded");
        assert_eq!(
            json["error"]["message"],
            "request did not start before the warm queue timeout"
        );

        let _ = release_tx.send(());
        held_slot.await.unwrap();
    }

    #[tokio::test]
    async fn ready_chat_request_times_out_in_shared_warm_queue_with_429() {
        let ready_status = BackendStatus {
            lifecycle: LifecycleState::Ready,
            tunnel: TunnelState::Ready,
            ..BackendStatus::default()
        };
        let config = AppConfig {
            warm_execution: WarmExecutionConfig {
                max_active_requests: 1,
                max_queued_requests: 1,
                queue_timeout: std::time::Duration::from_millis(20),
            },
            ..test_config()
        };
        let scheduler = WarmExecutionScheduler::new(config.warm_execution.clone());
        let (release_tx, release_rx) = tokio::sync::oneshot::channel::<()>();
        let held_slot = tokio::spawn({
            let scheduler = scheduler.clone();
            async move {
                scheduler
                    .execute(LifecycleRequest::Embeddings, |_| async move {
                        let _ = release_rx.await;
                    })
                    .await
                    .unwrap();
            }
        });
        tokio::task::yield_now().await;

        let app = build_router(AppState::with_services(
            config,
            Arc::new(StaticLifecycleOrchestrator {
                decision: LifecycleDecision::Ready(ready_status),
            }),
            scheduler,
        ));

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

        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["type"], "overloaded");

        let _ = release_tx.send(());
        held_slot.await.unwrap();
    }

    #[tokio::test]
    async fn ready_chat_request_hits_immediate_shared_queue_full_with_429() {
        let ready_status = BackendStatus {
            lifecycle: LifecycleState::Ready,
            tunnel: TunnelState::Ready,
            ..BackendStatus::default()
        };
        let config = AppConfig {
            warm_execution: WarmExecutionConfig {
                max_active_requests: 1,
                max_queued_requests: 0,
                queue_timeout: std::time::Duration::from_millis(20),
            },
            ..test_config()
        };
        let scheduler = WarmExecutionScheduler::new(config.warm_execution.clone());
        let (release_tx, release_rx) = tokio::sync::oneshot::channel::<()>();
        let held_slot = tokio::spawn({
            let scheduler = scheduler.clone();
            async move {
                scheduler
                    .execute(LifecycleRequest::Embeddings, |_| async move {
                        let _ = release_rx.await;
                    })
                    .await
                    .unwrap();
            }
        });
        tokio::task::yield_now().await;

        let app = build_router(AppState::with_services(
            config,
            Arc::new(StaticLifecycleOrchestrator {
                decision: LifecycleDecision::Ready(ready_status),
            }),
            scheduler,
        ));

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

        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["type"], "overloaded");
        assert_eq!(json["error"]["message"], "warm execution queue is full");

        let _ = release_tx.send(());
        held_slot.await.unwrap();
    }

    #[tokio::test]
    async fn status_embeddings_block_is_null_when_disabled() {
        let config = AppConfig {
            embeddings: EmbeddingsConfig {
                enabled: false,
                backend: None,
            },
            ..test_config()
        };
        let app = build_router(AppState::new(config));

        let response = app
            .oneshot(Request::get("/status").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json["embeddings"], serde_json::Value::Null);
    }

    #[tokio::test]
    async fn status_dual_mode_reports_independent_backends() {
        let chat_status = BackendStatus {
            lifecycle: LifecycleState::Ready,
            tunnel: TunnelState::Ready,
            ..BackendStatus::default()
        };
        let embeddings_status = BackendStatus {
            lifecycle: LifecycleState::Warming,
            tunnel: TunnelState::Connecting,
            ..BackendStatus::default()
        };
        let app = build_router(dual_mode_state(
            LifecycleDecision::Ready(chat_status),
            LifecycleDecision::Warming {
                status: embeddings_status,
                retry_after_secs: 10,
            },
        ));

        let response = app
            .oneshot(Request::get("/status").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(json["chat"]["model_alias"], "proxy-model");
        assert_eq!(json["chat"]["state"], "ready");
        assert_eq!(json["chat"]["tunnel"], "ready");

        assert_eq!(json["embeddings"]["model_alias"], "embed-model");
        assert_eq!(json["embeddings"]["state"], "warming");
        assert_eq!(json["embeddings"]["tunnel"], "connecting");
    }

    #[tokio::test]
    async fn models_lists_two_entries_in_dual_mode() {
        let app = build_router(dual_mode_state(
            LifecycleDecision::Ready(BackendStatus::default()),
            LifecycleDecision::Ready(BackendStatus::default()),
        ));

        let response = app
            .oneshot(Request::get("/v1/models").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();

        let data = json["data"].as_array().unwrap();
        assert_eq!(data.len(), 2);
        assert_eq!(data[0]["id"], "proxy-model");
        assert_eq!(data[1]["id"], "embed-model");
        assert_eq!(data[1]["owned_by"], "test-suite");
        assert_eq!(data[1]["provider"], "llama.cpp");
    }

    #[tokio::test]
    async fn embeddings_dual_mode_validates_against_dedicated_alias() {
        let app = build_router(dual_mode_state(
            LifecycleDecision::Ready(BackendStatus::default()),
            LifecycleDecision::Ready(BackendStatus::default()),
        ));

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
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["type"], "model_not_found");
        assert_eq!(
            json["error"]["message"],
            "unsupported model 'proxy-model': expected 'embed-model'"
        );
    }

    #[tokio::test]
    async fn dual_mode_routes_chat_and_embeddings_to_independent_lifecycles() {
        let ready_status = BackendStatus {
            lifecycle: LifecycleState::Ready,
            tunnel: TunnelState::Ready,
            ..BackendStatus::default()
        };
        let warming_status = BackendStatus {
            lifecycle: LifecycleState::Warming,
            tunnel: TunnelState::Down,
            ..BackendStatus::default()
        };

        // Chat ready, embeddings warming: each request must reflect its own
        // backend's lifecycle, not the other's.
        let app = build_router(dual_mode_state(
            LifecycleDecision::Ready(ready_status.clone()),
            LifecycleDecision::Warming {
                status: warming_status.clone(),
                retry_after_secs: 10,
            },
        ));

        let chat_response = app
            .clone()
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
        assert_eq!(chat_response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(chat_response.headers().get("retry-after").is_none());
        let body = chat_response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["type"], "backend_unavailable");

        let embeddings_response = app
            .oneshot(
                Request::post("/v1/embeddings")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"model":"embed-model","input":"hello"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            embeddings_response.status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert_eq!(embeddings_response.headers()["retry-after"], "10");
        let body = embeddings_response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["type"], "warming_up");

        // Reverse: chat warming, embeddings ready.
        let app = build_router(dual_mode_state(
            LifecycleDecision::Warming {
                status: warming_status,
                retry_after_secs: 10,
            },
            LifecycleDecision::Ready(ready_status),
        ));

        let chat_response = app
            .clone()
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
        assert_eq!(chat_response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(chat_response.headers()["retry-after"], "10");
        let body = chat_response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["type"], "warming_up");

        let embeddings_response = app
            .oneshot(
                Request::post("/v1/embeddings")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"model":"embed-model","input":"hello"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            embeddings_response.status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        assert!(embeddings_response.headers().get("retry-after").is_none());
        let body = embeddings_response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["error"]["type"], "backend_unavailable");
    }
}

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    http::request::Parts,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use axum::extract::{FromRequest, FromRequestParts, rejection::JsonRejection};
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

struct OpenAiJson<T>(T);

impl<S, T> FromRequest<S> for OpenAiJson<T>
where
    S: Send + Sync,
    Json<T>: FromRequest<S, Rejection = JsonRejection>,
{
    type Rejection = Response;

    async fn from_request(req: axum::extract::Request, state: &S) -> Result<Self, Self::Rejection> {
        match Json::<T>::from_request(req, state).await {
            Ok(Json(value)) => Ok(Self(value)),
            Err(rejection) => Err(json_rejection_to_response(rejection)),
        }
    }
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
    _: OptionalAuthorization,
    OpenAiJson(payload): OpenAiJson<ChatCompletionRequest>,
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
    _: OptionalAuthorization,
    OpenAiJson(payload): OpenAiJson<EmbeddingsRequest>,
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

fn json_rejection_to_response(rejection: JsonRejection) -> Response {
    let message = match rejection {
        JsonRejection::MissingJsonContentType(_) => {
            "content-type must be application/json".to_string()
        }
        JsonRejection::JsonSyntaxError(_) | JsonRejection::BytesRejection(_) => {
            "request body must contain valid JSON".to_string()
        }
        JsonRejection::JsonDataError(err) => normalize_json_data_error(err.body_text()),
        _ => "request body could not be parsed".to_string(),
    };

    openai_error(
        StatusCode::BAD_REQUEST,
        "invalid_request_error",
        message,
        None,
    )
}

fn normalize_json_data_error(message: String) -> String {
    if let Some(field) = extract_quoted_value(&message, "unknown field `") {
        return format!("unsupported field '{field}'");
    }

    if message.contains("unknown variant") {
        return "request body contains invalid fields".to_string();
    }

    if message.contains("missing field `tool_call_id`") {
        return "tool messages must include tool_call_id".to_string();
    }

    if message.contains("missing field `content`") {
        return "messages must include content unless assistant tool_calls are present"
            .to_string();
    }

    if message.contains("invalid type") {
        return "request body contains an invalid field type".to_string();
    }

    "request body contains invalid fields".to_string()
}

fn extract_quoted_value<'a>(message: &'a str, prefix: &str) -> Option<&'a str> {
    let suffix = &message[message.find(prefix)? + prefix.len()..];
    let end = suffix.find('`')?;
    Some(&suffix[..end])
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
    async fn chat_accepts_tool_calling_fields_and_null_assistant_content() {
        let state = test_state();
        let app = build_router(state.clone());

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

        let backend = state.backend.read().await;
        assert!(matches!(backend.lifecycle, LifecycleState::Warming));
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
    async fn chat_accepts_authorization_header_identically() {
        let state = test_state();
        let app = build_router(state.clone());

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

        let backend = state.backend.read().await;
        assert!(matches!(backend.lifecycle, LifecycleState::Warming));
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
        assert_eq!(json["error"]["message"], "tool messages must include tool_call_id");
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
        assert_eq!(json["error"]["message"], "request body contains an invalid field type");
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
        assert_eq!(json["error"]["message"], "request body contains invalid fields");
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
        assert_eq!(json["error"]["message"], "request body contains invalid fields");
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
        assert_eq!(json["error"]["message"], "request body contains invalid fields");
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
        assert_eq!(json["error"]["message"], "assistant messages tool_calls must not be empty");
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
        assert_eq!(json["error"]["message"], "only assistant messages may include tool_calls");
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
        assert_eq!(json["error"]["message"], "only tool messages may include tool_call_id");
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
        assert_eq!(json["error"]["message"], "content-type must be application/json");
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
        assert_eq!(json["error"]["message"], "unsupported field 'encoding_format'");
    }

    #[tokio::test]
    async fn embeddings_accepts_authorization_header_identically() {
        let state = test_state();
        let app = build_router(state.clone());

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

        let backend = state.backend.read().await;
        assert!(matches!(backend.lifecycle, LifecycleState::Warming));
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

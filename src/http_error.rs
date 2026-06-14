use axum::{
    Json,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use serde_json::{Value, json};

pub(crate) fn openai_error(
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

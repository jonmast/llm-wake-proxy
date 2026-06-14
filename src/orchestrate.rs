use axum::{body::Bytes, http::StatusCode, response::Response};
use tokio::sync::OwnedSemaphorePermit;

use crate::{
    forward::{self, ForwardConfig},
    http_error::openai_error,
    lifecycle::{BackendStatus, CapabilityState, LifecycleDecision, LifecycleRequest},
    scheduler::{RequestCancellation, WarmExecutionError},
    state::AppState,
};

pub struct Orchestrator {
    state: AppState,
}

impl Orchestrator {
    pub fn new(state: AppState) -> Self {
        Self { state }
    }

    pub async fn execute(&self, kind: LifecycleRequest, body: Bytes, stream: bool) -> Response {
        let _cold_permit = match self.acquire_cold_permit() {
            Ok(permit) => permit,
            Err(response) => return response,
        };

        match self.state.lifecycle.ensure_backend(kind.clone()).await {
            LifecycleDecision::Ready(status) => self.handle_ready(kind, body, stream, status).await,
            LifecycleDecision::Warming {
                retry_after_secs, ..
            } => {
                self.state.metrics.inc_cold_starts();
                self.state.metrics.inc_wake_attempts();
                openai_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "warming_up",
                    "backend is still starting".to_string(),
                    Some(&retry_after_secs.to_string()),
                )
            }
            LifecycleDecision::Failed { error, .. } => {
                self.state.metrics.inc_wake_failures();
                openai_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "backend_error",
                    error.message,
                    None,
                )
            }
        }
    }

    fn acquire_cold_permit(&self) -> Result<Option<OwnedSemaphorePermit>, Response> {
        if self.state.is_cold() {
            match self.state.cold_start_semaphore.clone().try_acquire_owned() {
                Ok(permit) => Ok(Some(permit)),
                Err(_) => Err(openai_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "warming_up",
                    "cold-start queue is full, retry later".to_string(),
                    Some("10"),
                )),
            }
        } else {
            Ok(None)
        }
    }

    async fn handle_ready(
        &self,
        kind: LifecycleRequest,
        body: Bytes,
        stream: bool,
        status: BackendStatus,
    ) -> Response {
        if self.state.check_tunnel_drop(status.tunnel) {
            self.state.metrics.inc_tunnel_drops();
        }

        if kind == LifecycleRequest::Embeddings && status.embeddings == CapabilityState::Degraded
        {
            return openai_error(
                StatusCode::BAD_REQUEST,
                "unsupported_embeddings",
                status
                    .embeddings_reason
                    .unwrap_or_else(|| "embeddings are degraded".to_string()),
                None,
            );
        }

        let path = match kind {
            LifecycleRequest::Chat => "/v1/chat/completions",
            LifecycleRequest::Embeddings => "/v1/embeddings",
        };

        self.forward_via_scheduler(kind, body, stream, path).await
    }

    async fn forward_via_scheduler(
        &self,
        kind: LifecycleRequest,
        body: Bytes,
        stream: bool,
        path: &str,
    ) -> Response {
        let state = &self.state;
        let config = ForwardConfig {
            port: state.config.host.tunnel_local_port,
        };
        let model_alias = state.config.model.alias.clone();

        let forward_body = body.clone();
        let forward_config = config.clone();
        let forward_model_alias = model_alias.clone();

        match state
            .scheduler
            .execute(kind.clone(), |cancellation| async move {
                if stream {
                    forward::forward_streaming(
                        &forward_config,
                        path,
                        forward_body,
                        &cancellation,
                        &forward_model_alias,
                    )
                    .await
                } else {
                    forward::forward_non_streaming(
                        &forward_config,
                        path,
                        forward_body,
                        &cancellation,
                        &forward_model_alias,
                    )
                    .await
                }
            })
            .await
        {
            Ok(Ok(response)) => {
                state.metrics.inc_warm_requests();
                response
            }
            Ok(Err(forward::ForwardError::EmbeddingsUnsupported)) => {
                state.metrics.inc_embeddings_degraded();
                embedding_degradation_response(state, "upstream returned 404").await
            }
            Ok(Err(forward::ForwardError::UpstreamUnreachable)) => {
                state.metrics.inc_cold_starts();
                state.metrics.inc_wake_attempts();
                state.lifecycle.mark_warming();
                match state.lifecycle.ensure_backend(kind).await {
                    LifecycleDecision::Ready(_) => {
                        let cancellation = RequestCancellation::new();
                        if stream {
                            forward::forward_streaming(
                                &config,
                                path,
                                body,
                                &cancellation,
                                &model_alias,
                            )
                            .await
                            .unwrap_or_else(|e| e.to_openai())
                        } else {
                            forward::forward_non_streaming(
                                &config,
                                path,
                                body,
                                &cancellation,
                                &model_alias,
                            )
                            .await
                            .unwrap_or_else(|e| e.to_openai())
                        }
                    }
                    LifecycleDecision::Warming {
                        retry_after_secs, ..
                    } => openai_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "warming_up",
                        "backend is still starting".to_string(),
                        Some(&retry_after_secs.to_string()),
                    ),
                    LifecycleDecision::Failed { error, .. } => openai_error(
                        StatusCode::SERVICE_UNAVAILABLE,
                        "backend_error",
                        error.message,
                        None,
                    ),
                }
            }
            Ok(Err(error)) => {
                state.metrics.inc_forwarding_errors();
                error.to_openai()
            }
            Err(WarmExecutionError::QueueFull) => {
                state.metrics.inc_queue_full();
                warm_execution_error_response(WarmExecutionError::QueueFull)
            }
            Err(WarmExecutionError::QueueTimeout) => {
                state.metrics.inc_queue_timeouts();
                warm_execution_error_response(WarmExecutionError::QueueTimeout)
            }
        }
    }
}

async fn embedding_degradation_response(state: &AppState, reason: &str) -> Response {
    state
        .lifecycle
        .degrade_embeddings(reason.to_string())
        .await;
    openai_error(
        StatusCode::BAD_REQUEST,
        "unsupported_embeddings",
        format!("upstream backend does not support embeddings: {reason}"),
        None,
    )
}

fn warm_execution_error_response(error: WarmExecutionError) -> Response {
    openai_error(
        StatusCode::TOO_MANY_REQUESTS,
        "overloaded",
        error.message().to_string(),
        None,
    )
}

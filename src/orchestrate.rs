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
        let _cold_permit = match self.acquire_cold_permit(&kind) {
            Ok(permit) => permit,
            Err(response) => return *response,
        };

        match self
            .state
            .lifecycle_for(&kind)
            .ensure_backend(kind.clone())
            .await
        {
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

    fn acquire_cold_permit(
        &self,
        kind: &LifecycleRequest,
    ) -> Result<Option<OwnedSemaphorePermit>, Box<Response>> {
        if self.state.is_cold(kind) {
            match self.state.cold_start_semaphore.clone().try_acquire_owned() {
                Ok(permit) => Ok(Some(permit)),
                Err(_) => Err(Box::new(openai_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "warming_up",
                    "cold-start queue is full, retry later".to_string(),
                    Some("10"),
                ))),
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
        if self.state.check_tunnel_drop(&kind, status.tunnel) {
            self.state.metrics.inc_tunnel_drops();
        }

        if kind == LifecycleRequest::Embeddings && status.embeddings == CapabilityState::Degraded {
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
            port: state.tunnel_port_for(&kind),
        };
        let model_alias = state.model_alias_for(&kind).to_string();

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
                let lifecycle = state.lifecycle_for(&kind);
                lifecycle.mark_warming();
                match lifecycle.ensure_backend(kind).await {
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
        .embeddings_lifecycle
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

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    use axum::{Router, routing::post};
    use http_body_util::BodyExt;

    use super::*;
    use crate::{
        config::{AppConfig, EmbeddingsConfig, HostConfig, ModelConfig, WarmExecutionConfig},
        lifecycle::{
            LifecycleError, LifecycleFuture, LifecycleOrchestrator, LifecycleState, TunnelState,
        },
        scheduler::WarmExecutionScheduler,
    };

    #[derive(Clone)]
    struct StaticLifecycleOrchestrator {
        decision: LifecycleDecision,
    }

    impl LifecycleOrchestrator for StaticLifecycleOrchestrator {
        fn ensure_backend(
            &self,
            _request: LifecycleRequest,
        ) -> LifecycleFuture<'_, LifecycleDecision> {
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

        fn degrade_embeddings(&self, _reason: String) -> LifecycleFuture<'_, ()> {
            Box::pin(async move {})
        }

        fn mark_warming(&self) {}
    }

    /// Returns successive decisions on each `ensure_backend` call, repeating
    /// the last one once the sequence is exhausted.
    struct SequenceLifecycleOrchestrator {
        decisions: Vec<LifecycleDecision>,
        index: AtomicUsize,
    }

    impl SequenceLifecycleOrchestrator {
        fn new(decisions: Vec<LifecycleDecision>) -> Self {
            Self {
                decisions,
                index: AtomicUsize::new(0),
            }
        }
    }

    impl LifecycleOrchestrator for SequenceLifecycleOrchestrator {
        fn ensure_backend(
            &self,
            _request: LifecycleRequest,
        ) -> LifecycleFuture<'_, LifecycleDecision> {
            let i = self.index.fetch_add(1, Ordering::SeqCst);
            let last = self.decisions.len() - 1;
            let decision = self.decisions[i.min(last)].clone();
            Box::pin(async move { decision })
        }

        fn status(&self) -> BackendStatus {
            match self.decisions.last().expect("at least one decision") {
                LifecycleDecision::Ready(status) => status.clone(),
                LifecycleDecision::Warming { status, .. } => status.clone(),
                LifecycleDecision::Failed { status, .. } => status.clone(),
            }
        }

        fn degrade_embeddings(&self, _reason: String) -> LifecycleFuture<'_, ()> {
            Box::pin(async move {})
        }

        fn mark_warming(&self) {}
    }

    fn test_config() -> AppConfig {
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

    fn ready_status() -> BackendStatus {
        BackendStatus {
            lifecycle: LifecycleState::Ready,
            tunnel: TunnelState::Ready,
            ..BackendStatus::default()
        }
    }

    fn state_with_lifecycle(decision: LifecycleDecision) -> AppState {
        AppState::with_lifecycle(
            test_config(),
            Arc::new(StaticLifecycleOrchestrator { decision }),
        )
    }

    async fn json_body(response: Response) -> serde_json::Value {
        let body = response.into_body().collect().await.unwrap().to_bytes();
        serde_json::from_slice(&body).unwrap()
    }

    /// Binds an axum router to an ephemeral localhost port and serves it in
    /// the background, returning the bound port.
    async fn spawn_backend(app: Router) -> u16 {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        port
    }

    #[tokio::test]
    async fn warming_decision_returns_warming_up_with_retry_after() {
        let backend = BackendStatus {
            lifecycle: LifecycleState::Warming,
            ..BackendStatus::default()
        };
        let state = state_with_lifecycle(LifecycleDecision::Warming {
            status: backend,
            retry_after_secs: 7,
        });

        let response = Orchestrator::new(state.clone())
            .execute(LifecycleRequest::Chat, Bytes::new(), false)
            .await;

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(response.headers().get("retry-after").unwrap(), "7");
        let json = json_body(response).await;
        assert_eq!(json["error"]["type"], "warming_up");

        let snapshot = state.metrics.snapshot();
        assert_eq!(snapshot.cold_starts, 1);
        assert_eq!(snapshot.wake_attempts, 1);
    }

    #[tokio::test]
    async fn failed_decision_returns_backend_error() {
        let backend = BackendStatus {
            lifecycle: LifecycleState::Error,
            ..BackendStatus::default()
        };
        let state = state_with_lifecycle(LifecycleDecision::Failed {
            status: backend,
            error: LifecycleError::new("helper command failed"),
        });

        let response = Orchestrator::new(state.clone())
            .execute(LifecycleRequest::Embeddings, Bytes::new(), false)
            .await;

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(response.headers().get("retry-after").is_none());
        let json = json_body(response).await;
        assert_eq!(json["error"]["type"], "backend_error");
        assert_eq!(json["error"]["message"], "helper command failed");

        assert_eq!(state.metrics.snapshot().wake_failures, 1);
    }

    #[tokio::test]
    async fn cold_start_semaphore_exhaustion_returns_warming_up() {
        let mut config = test_config();
        config.cold_start_max_waiting = 1;
        let backend = BackendStatus::default();

        let state = AppState::with_lifecycle(
            config,
            Arc::new(StaticLifecycleOrchestrator {
                decision: LifecycleDecision::Ready(backend),
            }),
        );

        let _permit = state
            .cold_start_semaphore
            .clone()
            .try_acquire_owned()
            .unwrap();

        let response = Orchestrator::new(state)
            .execute(LifecycleRequest::Chat, Bytes::new(), false)
            .await;

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(response.headers().get("retry-after").unwrap(), "10");
        let json = json_body(response).await;
        assert_eq!(json["error"]["type"], "warming_up");
        assert_eq!(
            json["error"]["message"],
            "cold-start queue is full, retry later"
        );
    }

    #[tokio::test]
    async fn embeddings_degraded_capability_returns_bad_request_without_forwarding() {
        let backend = BackendStatus {
            embeddings: CapabilityState::Degraded,
            embeddings_reason: Some("remote disabled".to_string()),
            ..ready_status()
        };
        let state = state_with_lifecycle(LifecycleDecision::Ready(backend));

        let response = Orchestrator::new(state)
            .execute(LifecycleRequest::Embeddings, Bytes::new(), false)
            .await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let json = json_body(response).await;
        assert_eq!(json["error"]["type"], "unsupported_embeddings");
        assert_eq!(json["error"]["message"], "remote disabled");
    }

    #[tokio::test]
    async fn ready_decision_forwards_and_rewrites_model_alias() {
        let app = Router::new().route(
            "/v1/chat/completions",
            post(|| async {
                (
                    StatusCode::OK,
                    [(axum::http::header::CONTENT_TYPE, "application/json")],
                    r#"{"model":"llama-3-8b","choices":[]}"#,
                )
            }),
        );
        let port = spawn_backend(app).await;

        let mut config = test_config();
        config.host.tunnel_local_port = port;
        let backend = ready_status();
        let state = AppState::with_lifecycle(
            config,
            Arc::new(StaticLifecycleOrchestrator {
                decision: LifecycleDecision::Ready(backend),
            }),
        );

        let response = Orchestrator::new(state.clone())
            .execute(
                LifecycleRequest::Chat,
                Bytes::from(r#"{"model":"proxy-model","messages":[]}"#),
                false,
            )
            .await;

        assert_eq!(response.status(), StatusCode::OK);
        let json = json_body(response).await;
        assert_eq!(json["model"], "proxy-model");
        assert_eq!(state.metrics.snapshot().warm_requests, 1);
    }

    #[tokio::test]
    async fn ready_decision_embeddings_404_triggers_degradation_response() {
        let app = Router::new().route("/v1/embeddings", post(|| async { StatusCode::NOT_FOUND }));
        let port = spawn_backend(app).await;

        let mut config = test_config();
        config.host.tunnel_local_port = port;
        let backend = ready_status();
        let state = AppState::with_lifecycle(
            config,
            Arc::new(StaticLifecycleOrchestrator {
                decision: LifecycleDecision::Ready(backend),
            }),
        );

        let response = Orchestrator::new(state.clone())
            .execute(
                LifecycleRequest::Embeddings,
                Bytes::from(r#"{"model":"proxy-model","input":"hi"}"#),
                false,
            )
            .await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let json = json_body(response).await;
        assert_eq!(json["error"]["type"], "unsupported_embeddings");
        assert_eq!(state.metrics.snapshot().embeddings_degraded, 1);
    }

    #[tokio::test]
    async fn ready_decision_forward_error_increments_forwarding_errors() {
        let app = Router::new().route(
            "/v1/chat/completions",
            post(|| async { (StatusCode::INTERNAL_SERVER_ERROR, "boom") }),
        );
        let port = spawn_backend(app).await;

        let mut config = test_config();
        config.host.tunnel_local_port = port;
        let backend = ready_status();
        let state = AppState::with_lifecycle(
            config,
            Arc::new(StaticLifecycleOrchestrator {
                decision: LifecycleDecision::Ready(backend),
            }),
        );

        let response = Orchestrator::new(state.clone())
            .execute(
                LifecycleRequest::Chat,
                Bytes::from(r#"{"model":"proxy-model","messages":[]}"#),
                false,
            )
            .await;

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let json = json_body(response).await;
        assert_eq!(json["error"]["type"], "upstream_error");
        assert_eq!(state.metrics.snapshot().forwarding_errors, 1);
    }

    #[tokio::test]
    async fn ready_decision_unreachable_backend_retries_and_returns_backend_unavailable() {
        let mut config = test_config();
        config.host.tunnel_local_port = 1; // nothing listens on port 1
        let backend = ready_status();
        let state = AppState::with_lifecycle(
            config,
            Arc::new(StaticLifecycleOrchestrator {
                decision: LifecycleDecision::Ready(backend),
            }),
        );

        let response = Orchestrator::new(state.clone())
            .execute(
                LifecycleRequest::Chat,
                Bytes::from(r#"{"model":"proxy-model","messages":[]}"#),
                false,
            )
            .await;

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let json = json_body(response).await;
        assert_eq!(json["error"]["type"], "backend_unavailable");

        let snapshot = state.metrics.snapshot();
        assert_eq!(snapshot.cold_starts, 1);
        assert_eq!(snapshot.wake_attempts, 1);
    }

    #[tokio::test]
    async fn ready_decision_retry_reports_warming() {
        let backend = ready_status();
        let mut config = test_config();
        config.host.tunnel_local_port = 1;

        let state = AppState::with_lifecycle(
            config,
            Arc::new(SequenceLifecycleOrchestrator::new(vec![
                LifecycleDecision::Ready(backend.clone()),
                LifecycleDecision::Warming {
                    status: backend,
                    retry_after_secs: 5,
                },
            ])),
        );

        let response = Orchestrator::new(state)
            .execute(
                LifecycleRequest::Chat,
                Bytes::from(r#"{"model":"proxy-model","messages":[]}"#),
                false,
            )
            .await;

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(response.headers().get("retry-after").unwrap(), "5");
        let json = json_body(response).await;
        assert_eq!(json["error"]["type"], "warming_up");
    }

    #[tokio::test]
    async fn ready_decision_retry_reports_failed() {
        let backend = ready_status();
        let mut config = test_config();
        config.host.tunnel_local_port = 1;

        let state = AppState::with_lifecycle(
            config,
            Arc::new(SequenceLifecycleOrchestrator::new(vec![
                LifecycleDecision::Ready(backend.clone()),
                LifecycleDecision::Failed {
                    status: backend,
                    error: LifecycleError::new("ssh unreachable"),
                },
            ])),
        );

        let response = Orchestrator::new(state)
            .execute(
                LifecycleRequest::Chat,
                Bytes::from(r#"{"model":"proxy-model","messages":[]}"#),
                false,
            )
            .await;

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let json = json_body(response).await;
        assert_eq!(json["error"]["type"], "backend_error");
        assert_eq!(json["error"]["message"], "ssh unreachable");
    }

    #[tokio::test]
    async fn tunnel_drop_increments_metric() {
        let mut config = test_config();
        config.host.tunnel_local_port = 1;

        let ready = ready_status();
        let dropped = BackendStatus {
            tunnel: TunnelState::Down,
            ..ready_status()
        };

        let state = AppState::with_lifecycle(
            config,
            Arc::new(SequenceLifecycleOrchestrator::new(vec![
                LifecycleDecision::Ready(ready),
                LifecycleDecision::Ready(dropped),
            ])),
        );

        let orchestrator = Orchestrator::new(state.clone());
        orchestrator
            .execute(LifecycleRequest::Chat, Bytes::new(), false)
            .await;
        assert_eq!(state.metrics.snapshot().tunnel_drops, 0);

        orchestrator
            .execute(LifecycleRequest::Chat, Bytes::new(), false)
            .await;
        assert_eq!(state.metrics.snapshot().tunnel_drops, 1);
    }

    #[tokio::test]
    async fn warm_queue_full_returns_overloaded_429() {
        let backend = ready_status();
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
        let held = tokio::spawn({
            let scheduler = scheduler.clone();
            async move {
                scheduler
                    .execute(LifecycleRequest::Chat, |_| async move {
                        let _ = release_rx.await;
                    })
                    .await
                    .unwrap()
            }
        });
        tokio::task::yield_now().await;

        let state = AppState::with_services(
            config,
            Arc::new(StaticLifecycleOrchestrator {
                decision: LifecycleDecision::Ready(backend),
            }),
            scheduler,
        );

        let response = Orchestrator::new(state.clone())
            .execute(LifecycleRequest::Chat, Bytes::new(), false)
            .await;

        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        let json = json_body(response).await;
        assert_eq!(json["error"]["type"], "overloaded");
        assert_eq!(state.metrics.snapshot().queue_full_rejections, 1);

        let _ = release_tx.send(());
        held.await.unwrap();
    }

    #[tokio::test]
    async fn warm_queue_timeout_returns_overloaded_429() {
        let backend = ready_status();
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
        let held = tokio::spawn({
            let scheduler = scheduler.clone();
            async move {
                scheduler
                    .execute(LifecycleRequest::Chat, |_| async move {
                        let _ = release_rx.await;
                    })
                    .await
                    .unwrap()
            }
        });
        tokio::task::yield_now().await;

        let state = AppState::with_services(
            config,
            Arc::new(StaticLifecycleOrchestrator {
                decision: LifecycleDecision::Ready(backend),
            }),
            scheduler,
        );

        let response = Orchestrator::new(state.clone())
            .execute(LifecycleRequest::Chat, Bytes::new(), false)
            .await;

        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
        let json = json_body(response).await;
        assert_eq!(json["error"]["type"], "overloaded");
        assert_eq!(state.metrics.snapshot().queue_timeouts, 1);

        let _ = release_tx.send(());
        held.await.unwrap();
    }
}

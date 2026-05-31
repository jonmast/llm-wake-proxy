use std::{future::Future, pin::Pin};

use serde::Serialize;
use tokio::sync::Mutex;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct Timestamp(u64);

impl Timestamp {
    pub fn new(unix_seconds: u64) -> Self {
        Self(unix_seconds)
    }
}

pub type LifecycleFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LifecycleState {
    Cold,
    Warming,
    Ready,
    Error,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityState {
    Ready,
    Degraded,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TunnelState {
    Down,
    Connecting,
    Ready,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UnitState {
    Unknown,
    Inactive,
    Activating,
    Active,
    Failed,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LifecycleRequest {
    Chat,
    Embeddings,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LifecycleError {
    pub message: String,
}

impl LifecycleError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct BackendStatus {
    pub lifecycle: LifecycleState,
    pub chat: CapabilityState,
    pub embeddings: CapabilityState,
    pub embeddings_reason: Option<String>,
    pub tunnel: TunnelState,
    pub last_wake_attempt_at: Option<Timestamp>,
    pub lease_expires_at: Option<Timestamp>,
    pub llama_server_unit: UnitState,
    pub inhibit_unit: UnitState,
}

impl Default for BackendStatus {
    fn default() -> Self {
        Self {
            lifecycle: LifecycleState::Cold,
            chat: CapabilityState::Ready,
            embeddings: CapabilityState::Ready,
            embeddings_reason: None,
            tunnel: TunnelState::Down,
            last_wake_attempt_at: None,
            lease_expires_at: None,
            llama_server_unit: UnitState::Unknown,
            inhibit_unit: UnitState::Unknown,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObservedBackendState {
    pub lifecycle: LifecycleState,
    pub chat: CapabilityState,
    pub embeddings: CapabilityState,
    pub embeddings_reason: Option<String>,
    pub error: Option<String>,
    pub llama_server_unit: UnitState,
    pub inhibit_unit: UnitState,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LifecycleDecision {
    Ready(BackendStatus),
    Warming {
        status: BackendStatus,
        retry_after_secs: u64,
    },
    Failed {
        status: BackendStatus,
        error: LifecycleError,
    },
}

pub trait WakeRequester: Send + Sync {
    fn request_wake(
        &self,
        request: &LifecycleRequest,
    ) -> LifecycleFuture<'_, Result<(), LifecycleError>>;
}

pub trait SshReadinessProbe: Send + Sync {
    fn is_ready(&self) -> LifecycleFuture<'_, Result<bool, LifecycleError>>;
}

pub trait HelperRpc: Send + Sync {
    fn observe_backend(
        &self,
        request: &LifecycleRequest,
    ) -> LifecycleFuture<'_, Result<ObservedBackendState, LifecycleError>>;
}

pub trait TunnelOwner: Send + Sync {
    fn ensure_tunnel(&self) -> LifecycleFuture<'_, Result<TunnelState, LifecycleError>>;
}

pub trait Clock: Send + Sync {
    fn now(&self) -> Timestamp;
}

pub trait BackendStatePublisher: Send + Sync {
    fn snapshot(&self) -> BackendStatus;
    fn publish(&self, status: BackendStatus);
}

pub trait LifecycleOrchestrator: Send + Sync {
    fn ensure_backend(&self, request: LifecycleRequest) -> LifecycleFuture<'_, LifecycleDecision>;
    fn status(&self) -> BackendStatus;
}

pub struct LifecycleManager<W, S, H, T, C, P> {
    wake: W,
    ssh: S,
    helper: H,
    tunnel: T,
    clock: C,
    state: P,
    retry_after_secs: u64,
    orchestration_lock: Mutex<()>,
}

impl<W, S, H, T, C, P> LifecycleManager<W, S, H, T, C, P> {
    pub fn new(wake: W, ssh: S, helper: H, tunnel: T, clock: C, state: P) -> Self {
        Self {
            wake,
            ssh,
            helper,
            tunnel,
            clock,
            state,
            retry_after_secs: 10,
            orchestration_lock: Mutex::new(()),
        }
    }
}

impl<W, S, H, T, C, P> LifecycleOrchestrator for LifecycleManager<W, S, H, T, C, P>
where
    W: WakeRequester,
    S: SshReadinessProbe,
    H: HelperRpc,
    T: TunnelOwner,
    C: Clock,
    P: BackendStatePublisher,
{
    fn ensure_backend(&self, request: LifecycleRequest) -> LifecycleFuture<'_, LifecycleDecision> {
        Box::pin(async move {
            let _guard = self.orchestration_lock.lock().await;
            let mut status = self.state.snapshot();

            let ssh_ready = match self.ssh.is_ready().await {
                Ok(ready) => ready,
                Err(error) => {
                    status.lifecycle = LifecycleState::Error;
                    self.state.publish(status.clone());
                    return LifecycleDecision::Failed { status, error };
                }
            };

            if !ssh_ready {
                if should_request_wake(status.lifecycle) {
                    status.last_wake_attempt_at = Some(self.clock.now());
                    status.tunnel = TunnelState::Down;

                    if let Err(error) = self.wake.request_wake(&request).await {
                        status.lifecycle = LifecycleState::Error;
                        self.state.publish(status.clone());
                        return LifecycleDecision::Failed { status, error };
                    }
                }

                status.lifecycle = LifecycleState::Warming;
                status.tunnel = TunnelState::Down;
                self.state.publish(status.clone());
                return LifecycleDecision::Warming {
                    status,
                    retry_after_secs: self.retry_after_secs,
                };
            }

            let observed = match self.helper.observe_backend(&request).await {
                Ok(observed) => observed,
                Err(error) => {
                    status.lifecycle = LifecycleState::Error;
                    status.tunnel = TunnelState::Down;
                    self.state.publish(status.clone());
                    return LifecycleDecision::Failed { status, error };
                }
            };

            status.lifecycle = observed.lifecycle;
            status.chat = observed.chat;
            status.embeddings = observed.embeddings;
            status.embeddings_reason = observed.embeddings_reason;
            status.llama_server_unit = observed.llama_server_unit;
            status.inhibit_unit = observed.inhibit_unit;

            if matches!(status.lifecycle, LifecycleState::Error) {
                status.tunnel = TunnelState::Down;
                let error = LifecycleError::new(
                    observed
                        .error
                        .unwrap_or_else(|| "backend reported error state".to_string()),
                );
                self.state.publish(status.clone());
                return LifecycleDecision::Failed { status, error };
            }

            if !matches!(status.lifecycle, LifecycleState::Ready) {
                status.tunnel = TunnelState::Down;
                self.state.publish(status.clone());
                return LifecycleDecision::Warming {
                    status,
                    retry_after_secs: self.retry_after_secs,
                };
            }

            status.tunnel = match self.tunnel.ensure_tunnel().await {
                Ok(tunnel_state) => tunnel_state,
                Err(error) => {
                    status.lifecycle = LifecycleState::Error;
                    status.tunnel = TunnelState::Down;
                    self.state.publish(status.clone());
                    return LifecycleDecision::Failed { status, error };
                }
            };

            if !matches!(status.tunnel, TunnelState::Ready) {
                status.lifecycle = LifecycleState::Warming;
            }

            self.state.publish(status.clone());

            if matches!(status.lifecycle, LifecycleState::Ready)
                && matches!(status.tunnel, TunnelState::Ready)
            {
                LifecycleDecision::Ready(status)
            } else {
                LifecycleDecision::Warming {
                    status,
                    retry_after_secs: self.retry_after_secs,
                }
            }
        })
    }

    fn status(&self) -> BackendStatus {
        self.state.snapshot()
    }
}

fn should_request_wake(lifecycle: LifecycleState) -> bool {
    matches!(
        lifecycle,
        LifecycleState::Cold | LifecycleState::Ready | LifecycleState::Error
    )
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::*;

    const FAKE_NOW: u64 = 1_717_156_800;

    #[derive(Clone, Default)]
    struct FakeWakeRequester {
        calls: std::sync::Arc<Mutex<Vec<LifecycleRequest>>>,
        error: Option<LifecycleError>,
    }

    impl FakeWakeRequester {
        fn with_error(message: &str) -> Self {
            Self {
                calls: std::sync::Arc::new(Mutex::new(Vec::new())),
                error: Some(LifecycleError::new(message)),
            }
        }

        fn call_count(&self) -> usize {
            self.calls.lock().unwrap().len()
        }
    }

    impl WakeRequester for FakeWakeRequester {
        fn request_wake(
            &self,
            request: &LifecycleRequest,
        ) -> LifecycleFuture<'_, Result<(), LifecycleError>> {
            self.calls.lock().unwrap().push(request.clone());
            let error = self.error.clone();
            Box::pin(async move {
                match error {
                    Some(error) => Err(error),
                    None => Ok(()),
                }
            })
        }
    }

    struct FakeSshReadinessProbe {
        ready: bool,
    }

    impl SshReadinessProbe for FakeSshReadinessProbe {
        fn is_ready(&self) -> LifecycleFuture<'_, Result<bool, LifecycleError>> {
            let ready = self.ready;
            Box::pin(async move { Ok(ready) })
        }
    }

    #[derive(Default)]
    struct FakeHelperRpc {
        calls: Mutex<Vec<LifecycleRequest>>,
        observed: Option<ObservedBackendState>,
    }

    impl FakeHelperRpc {
        fn ready() -> Self {
            Self {
                calls: Mutex::new(Vec::new()),
                observed: Some(ObservedBackendState {
                    lifecycle: LifecycleState::Ready,
                    chat: CapabilityState::Ready,
                    embeddings: CapabilityState::Ready,
                    embeddings_reason: None,
                    error: None,
                    llama_server_unit: UnitState::Active,
                    inhibit_unit: UnitState::Active,
                }),
            }
        }

        fn error(message: &str) -> Self {
            Self {
                calls: Mutex::new(Vec::new()),
                observed: Some(ObservedBackendState {
                    lifecycle: LifecycleState::Error,
                    chat: CapabilityState::Ready,
                    embeddings: CapabilityState::Ready,
                    embeddings_reason: None,
                    error: Some(message.to_string()),
                    llama_server_unit: UnitState::Failed,
                    inhibit_unit: UnitState::Unknown,
                }),
            }
        }
    }

    impl HelperRpc for FakeHelperRpc {
        fn observe_backend(
            &self,
            request: &LifecycleRequest,
        ) -> LifecycleFuture<'_, Result<ObservedBackendState, LifecycleError>> {
            self.calls.lock().unwrap().push(request.clone());
            let observed = self
                .observed
                .clone()
                .expect("fake helper should have observed state");
            Box::pin(async move { Ok(observed) })
        }
    }

    #[derive(Clone)]
    struct FakeTunnelOwner {
        tunnel_state: TunnelState,
        calls: Arc<Mutex<usize>>,
    }

    impl FakeTunnelOwner {
        fn new(tunnel_state: TunnelState) -> Self {
            Self {
                tunnel_state,
                calls: Arc::new(Mutex::new(0)),
            }
        }

        fn call_count(&self) -> usize {
            *self.calls.lock().unwrap()
        }
    }

    impl TunnelOwner for FakeTunnelOwner {
        fn ensure_tunnel(&self) -> LifecycleFuture<'_, Result<TunnelState, LifecycleError>> {
            let tunnel_state = self.tunnel_state;
            *self.calls.lock().unwrap() += 1;
            Box::pin(async move { Ok(tunnel_state) })
        }
    }

    struct FakeClock {
        now: Timestamp,
    }

    impl Clock for FakeClock {
        fn now(&self) -> Timestamp {
            self.now.clone()
        }
    }

    #[derive(Clone)]
    struct FakeBackendStatePublisher {
        status: Arc<Mutex<BackendStatus>>,
    }

    impl FakeBackendStatePublisher {
        fn new(status: BackendStatus) -> Self {
            Self {
                status: Arc::new(Mutex::new(status)),
            }
        }

        fn latest(&self) -> BackendStatus {
            self.status.lock().unwrap().clone()
        }
    }

    impl BackendStatePublisher for FakeBackendStatePublisher {
        fn snapshot(&self) -> BackendStatus {
            self.latest()
        }

        fn publish(&self, status: BackendStatus) {
            *self.status.lock().unwrap() = status;
        }
    }

    #[tokio::test]
    async fn cold_request_marks_backend_warming_using_only_fakes() {
        let wake = FakeWakeRequester::default();
        let helper = FakeHelperRpc::default();
        let state = FakeBackendStatePublisher::new(BackendStatus::default());
        let lifecycle = LifecycleManager::new(
            wake,
            FakeSshReadinessProbe { ready: false },
            helper,
            FakeTunnelOwner::new(TunnelState::Down),
            FakeClock {
                now: Timestamp::new(FAKE_NOW),
            },
            state.clone(),
        );

        let decision = lifecycle.ensure_backend(LifecycleRequest::Chat).await;

        match decision {
            LifecycleDecision::Warming {
                status,
                retry_after_secs,
            } => {
                assert_eq!(retry_after_secs, 10);
                assert_eq!(status.lifecycle, LifecycleState::Warming);
                assert_eq!(status.last_wake_attempt_at, Some(Timestamp::new(FAKE_NOW)));
            }
            other => panic!("expected warming decision, got {other:?}"),
        }

        let latest = state.latest();
        assert_eq!(latest.lifecycle, LifecycleState::Warming);
        assert_eq!(latest.last_wake_attempt_at, Some(Timestamp::new(FAKE_NOW)));
    }

    #[tokio::test]
    async fn ready_backend_can_be_reached_with_fake_ports_only() {
        let state = FakeBackendStatePublisher::new(BackendStatus::default());
        let lifecycle = LifecycleManager::new(
            FakeWakeRequester::default(),
            FakeSshReadinessProbe { ready: true },
            FakeHelperRpc::ready(),
            FakeTunnelOwner::new(TunnelState::Ready),
            FakeClock {
                now: Timestamp::new(FAKE_NOW),
            },
            state.clone(),
        );

        let decision = lifecycle.ensure_backend(LifecycleRequest::Embeddings).await;

        match decision {
            LifecycleDecision::Ready(status) => {
                assert_eq!(status.lifecycle, LifecycleState::Ready);
                assert_eq!(status.tunnel, TunnelState::Ready);
                assert_eq!(status.llama_server_unit, UnitState::Active);
                assert_eq!(status.inhibit_unit, UnitState::Active);
            }
            other => panic!("expected ready decision, got {other:?}"),
        }

        let latest = state.latest();
        assert_eq!(latest.lifecycle, LifecycleState::Ready);
        assert_eq!(latest.tunnel, TunnelState::Ready);
    }

    #[tokio::test]
    async fn wake_failures_publish_error_without_touching_real_dependencies() {
        let state = FakeBackendStatePublisher::new(BackendStatus::default());
        let lifecycle = LifecycleManager::new(
            FakeWakeRequester::with_error("wake failed"),
            FakeSshReadinessProbe { ready: false },
            FakeHelperRpc::default(),
            FakeTunnelOwner::new(TunnelState::Down),
            FakeClock {
                now: Timestamp::new(FAKE_NOW),
            },
            state.clone(),
        );

        let decision = lifecycle.ensure_backend(LifecycleRequest::Chat).await;

        match decision {
            LifecycleDecision::Failed { status, error } => {
                assert_eq!(status.lifecycle, LifecycleState::Error);
                assert_eq!(error.message, "wake failed");
                assert_eq!(status.last_wake_attempt_at, Some(Timestamp::new(FAKE_NOW)));
            }
            other => panic!("expected failed decision, got {other:?}"),
        }

        let latest = state.latest();
        assert_eq!(latest.lifecycle, LifecycleState::Error);
        assert_eq!(latest.last_wake_attempt_at, Some(Timestamp::new(FAKE_NOW)));
    }

    #[tokio::test]
    async fn concurrent_cold_requests_share_a_single_wake() {
        let wake = FakeWakeRequester::default();
        let lifecycle = std::sync::Arc::new(LifecycleManager::new(
            wake.clone(),
            FakeSshReadinessProbe { ready: false },
            FakeHelperRpc::default(),
            FakeTunnelOwner::new(TunnelState::Down),
            FakeClock {
                now: Timestamp::new(FAKE_NOW),
            },
            FakeBackendStatePublisher::new(BackendStatus::default()),
        ));

        let (first, second) = tokio::join!(
            lifecycle.ensure_backend(LifecycleRequest::Chat),
            lifecycle.ensure_backend(LifecycleRequest::Chat)
        );

        assert!(matches!(first, LifecycleDecision::Warming { .. }));
        assert!(matches!(second, LifecycleDecision::Warming { .. }));

        assert_eq!(wake.call_count(), 1);
    }

    #[tokio::test]
    async fn helper_reported_error_returns_failed_decision() {
        let state = FakeBackendStatePublisher::new(BackendStatus::default());
        let lifecycle = LifecycleManager::new(
            FakeWakeRequester::default(),
            FakeSshReadinessProbe { ready: true },
            FakeHelperRpc::error("helper reported backend failure"),
            FakeTunnelOwner::new(TunnelState::Ready),
            FakeClock {
                now: Timestamp::new(FAKE_NOW),
            },
            state.clone(),
        );

        let decision = lifecycle.ensure_backend(LifecycleRequest::Chat).await;

        match decision {
            LifecycleDecision::Failed { status, error } => {
                assert_eq!(status.lifecycle, LifecycleState::Error);
                assert_eq!(status.tunnel, TunnelState::Down);
                assert_eq!(error.message, "helper reported backend failure");
            }
            other => panic!("expected failed decision, got {other:?}"),
        }

        let latest = state.latest();
        assert_eq!(latest.lifecycle, LifecycleState::Error);
        assert_eq!(latest.tunnel, TunnelState::Down);
    }

    #[tokio::test]
    async fn ssh_not_ready_republishes_warming_state_from_ready_snapshot() {
        let wake = FakeWakeRequester::default();
        let state = FakeBackendStatePublisher::new(BackendStatus {
            lifecycle: LifecycleState::Ready,
            tunnel: TunnelState::Ready,
            ..BackendStatus::default()
        });
        let lifecycle = LifecycleManager::new(
            wake.clone(),
            FakeSshReadinessProbe { ready: false },
            FakeHelperRpc::default(),
            FakeTunnelOwner::new(TunnelState::Ready),
            FakeClock {
                now: Timestamp::new(FAKE_NOW),
            },
            state.clone(),
        );

        let decision = lifecycle.ensure_backend(LifecycleRequest::Chat).await;

        match decision {
            LifecycleDecision::Warming { status, .. } => {
                assert_eq!(status.lifecycle, LifecycleState::Warming);
                assert_eq!(status.tunnel, TunnelState::Down);
                assert_eq!(status.last_wake_attempt_at, Some(Timestamp::new(FAKE_NOW)));
            }
            other => panic!("expected warming decision, got {other:?}"),
        }

        let latest = state.latest();
        assert_eq!(latest.lifecycle, LifecycleState::Warming);
        assert_eq!(latest.tunnel, TunnelState::Down);
        assert_eq!(wake.call_count(), 1);
    }

    #[tokio::test]
    async fn error_snapshot_retries_wake_when_ssh_is_not_ready() {
        let wake = FakeWakeRequester::default();
        let state = FakeBackendStatePublisher::new(BackendStatus {
            lifecycle: LifecycleState::Error,
            tunnel: TunnelState::Down,
            ..BackendStatus::default()
        });
        let lifecycle = LifecycleManager::new(
            wake.clone(),
            FakeSshReadinessProbe { ready: false },
            FakeHelperRpc::default(),
            FakeTunnelOwner::new(TunnelState::Down),
            FakeClock {
                now: Timestamp::new(FAKE_NOW),
            },
            state.clone(),
        );

        let decision = lifecycle.ensure_backend(LifecycleRequest::Chat).await;

        match decision {
            LifecycleDecision::Warming { status, .. } => {
                assert_eq!(status.lifecycle, LifecycleState::Warming);
                assert_eq!(status.tunnel, TunnelState::Down);
            }
            other => panic!("expected warming decision, got {other:?}"),
        }

        assert_eq!(state.latest().lifecycle, LifecycleState::Warming);
        assert_eq!(wake.call_count(), 1);
    }

    #[tokio::test]
    async fn warm_backend_reuses_existing_readiness_without_requesting_wake() {
        let wake = FakeWakeRequester::default();
        let helper = FakeHelperRpc::ready();
        let tunnel = FakeTunnelOwner::new(TunnelState::Ready);
        let state = FakeBackendStatePublisher::new(BackendStatus {
            lifecycle: LifecycleState::Ready,
            tunnel: TunnelState::Ready,
            ..BackendStatus::default()
        });
        let lifecycle = LifecycleManager::new(
            wake.clone(),
            FakeSshReadinessProbe { ready: true },
            helper,
            tunnel.clone(),
            FakeClock {
                now: Timestamp::new(FAKE_NOW),
            },
            state.clone(),
        );

        let decision = lifecycle.ensure_backend(LifecycleRequest::Chat).await;

        match decision {
            LifecycleDecision::Ready(status) => {
                assert_eq!(status.lifecycle, LifecycleState::Ready);
                assert_eq!(status.tunnel, TunnelState::Ready);
            }
            other => panic!("expected ready decision, got {other:?}"),
        }

        assert_eq!(wake.call_count(), 0);
        assert_eq!(tunnel.call_count(), 1);
        assert_eq!(state.latest().lifecycle, LifecycleState::Ready);
    }

    #[tokio::test]
    async fn restart_rediscovery_adopts_ready_backend_and_creates_fresh_tunnel() {
        let wake = FakeWakeRequester::default();
        let helper = FakeHelperRpc::ready();
        let tunnel = FakeTunnelOwner::new(TunnelState::Ready);
        let state = FakeBackendStatePublisher::new(BackendStatus::default());
        let lifecycle = LifecycleManager::new(
            wake.clone(),
            FakeSshReadinessProbe { ready: true },
            helper,
            tunnel.clone(),
            FakeClock {
                now: Timestamp::new(FAKE_NOW),
            },
            state.clone(),
        );

        let decision = lifecycle.ensure_backend(LifecycleRequest::Chat).await;

        match decision {
            LifecycleDecision::Ready(status) => {
                assert_eq!(status.lifecycle, LifecycleState::Ready);
                assert_eq!(status.tunnel, TunnelState::Ready);
                assert_eq!(status.llama_server_unit, UnitState::Active);
                assert_eq!(status.inhibit_unit, UnitState::Active);
            }
            other => panic!("expected ready decision, got {other:?}"),
        }

        assert_eq!(wake.call_count(), 0);
        assert_eq!(tunnel.call_count(), 1);
        assert_eq!(state.latest().lifecycle, LifecycleState::Ready);
    }
}

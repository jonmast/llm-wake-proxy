use std::{future::Future, pin::Pin, sync::Arc, time::Duration};

use serde::Serialize;
use tokio::{
    sync::{Mutex, watch},
    time::Instant,
};
use tracing::{debug, info, warn};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(transparent)]
pub struct Timestamp(u64);

impl Timestamp {
    pub fn new(unix_seconds: u64) -> Self {
        Self(unix_seconds)
    }

    pub fn unix_seconds(self) -> u64 {
        self.0
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

impl std::fmt::Display for LifecycleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
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

pub trait LifecycleOrchestrator: Send + Sync {
    fn ensure_backend(&self, request: LifecycleRequest) -> LifecycleFuture<'_, LifecycleDecision>;
    fn status(&self) -> BackendStatus;
    fn degrade_embeddings(&self, reason: String) -> LifecycleFuture<'_, ()>;
    fn mark_warming(&self);
}

#[derive(Clone, Copy, Debug)]
pub struct LifecycleTiming {
    pub cold_wait_budget_secs: u64,
    pub hard_boot_deadline_secs: u64,
    pub bootstrap_poll_interval_millis: u64,
    pub retry_after_secs: u64,
}

impl Default for LifecycleTiming {
    fn default() -> Self {
        Self {
            cold_wait_budget_secs: read_env_or("COLD_WAIT_BUDGET_SECS", 90),
            hard_boot_deadline_secs: read_env_or("HARD_BOOT_DEADLINE_SECS", 300),
            bootstrap_poll_interval_millis: read_env_or("BOOTSTRAP_POLL_INTERVAL_MS", 1_000),
            retry_after_secs: read_env_or("RETRY_AFTER_SECS", 10),
        }
    }
}

fn read_env_or(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

struct BootstrapState {
    active: bool,
    last_error: Option<LifecycleError>,
    observe_request: LifecycleRequest,
}

impl Default for BootstrapState {
    fn default() -> Self {
        Self {
            active: false,
            last_error: None,
            observe_request: LifecycleRequest::Chat,
        }
    }
}

struct LifecycleCore<W, S, H, T, C> {
    wake: W,
    ssh: S,
    helper: H,
    tunnel: T,
    clock: C,
    timing: LifecycleTiming,
    bootstrap: Mutex<BootstrapState>,
    status_tx: watch::Sender<BackendStatus>,
}

pub struct LifecycleManager<W, S, H, T, C> {
    core: Arc<LifecycleCore<W, S, H, T, C>>,
}

impl<W, S, H, T, C> LifecycleManager<W, S, H, T, C> {
    pub fn new(wake: W, ssh: S, helper: H, tunnel: T, clock: C, initial_status: BackendStatus) -> Self {
        Self::new_with_timing(
            wake,
            ssh,
            helper,
            tunnel,
            clock,
            initial_status,
            LifecycleTiming::default(),
        )
    }

    pub fn new_with_timing(
        wake: W,
        ssh: S,
        helper: H,
        tunnel: T,
        clock: C,
        initial_status: BackendStatus,
        timing: LifecycleTiming,
    ) -> Self {
        let (status_tx, _) = watch::channel(initial_status);

        Self {
            core: Arc::new(LifecycleCore {
                wake,
                ssh,
                helper,
                tunnel,
                clock,
                timing,
                bootstrap: Mutex::new(BootstrapState::default()),
                status_tx,
            }),
        }
    }
}

impl<W, S, H, T, C> LifecycleCore<W, S, H, T, C>
where
    W: WakeRequester + Send + Sync + 'static,
    S: SshReadinessProbe + Send + Sync + 'static,
    H: HelperRpc + Send + Sync + 'static,
    T: TunnelOwner + Send + Sync + 'static,
    C: Clock + Send + Sync + 'static,
{
    fn publish(&self, status: BackendStatus) {
        self.status_tx.send_replace(status);
    }

    async fn start_or_join_bootstrap(
        &self,
        initial_status: BackendStatus,
        request: &LifecycleRequest,
    ) -> (bool, watch::Receiver<BackendStatus>, BackendStatus) {
        let mut bootstrap = self.bootstrap.lock().await;
        let should_start = !bootstrap.active;

        if should_start {
            bootstrap.active = true;
            bootstrap.last_error = None;
            bootstrap.observe_request = request.clone();
        } else if request_priority(request) > request_priority(&bootstrap.observe_request) {
            bootstrap.observe_request = request.clone();
        }

        (should_start, self.status_tx.subscribe(), initial_status)
    }

    async fn bootstrap_observe_request(&self) -> LifecycleRequest {
        self.bootstrap.lock().await.observe_request.clone()
    }

    async fn observe_once(
        &self,
        wake_request: Option<&LifecycleRequest>,
        observe_request: LifecycleRequest,
        mut status: BackendStatus,
    ) -> LifecycleDecision {
        let ssh_ready = match self.ssh.is_ready().await {
            Ok(ready) => {
                debug!(ssh_ready = ready, "SSH readiness probe completed");
                ready
            }
            Err(error) => {
                warn!(error = %error, "SSH readiness probe failed");
                status.lifecycle = LifecycleState::Error;
                status.tunnel = TunnelState::Down;
                return LifecycleDecision::Failed { status, error };
            }
        };

        if !ssh_ready {
            if let Some(wake_request) =
                wake_request.filter(|_| should_request_wake(status.lifecycle))
            {
                info!(request = ?wake_request, "sending wake request");
                status.last_wake_attempt_at = Some(self.clock.now());
                status.tunnel = TunnelState::Down;

                if let Err(error) = self.wake.request_wake(wake_request).await {
                    warn!(error = %error, "wake request failed");
                    status.lifecycle = LifecycleState::Error;
                    return LifecycleDecision::Failed { status, error };
                }
            }

            status.lifecycle = LifecycleState::Warming;
            status.tunnel = TunnelState::Down;
            self.publish(status.clone());
            return LifecycleDecision::Warming {
                status,
                retry_after_secs: self.timing.retry_after_secs,
            };
        }

        let observed = match self.helper.observe_backend(&observe_request).await {
            Ok(observed) => {
                debug!(
                    lifecycle = ?observed.lifecycle,
                    chat = ?observed.chat,
                    embeddings = ?observed.embeddings,
                    "backend observation completed"
                );
                observed
            }
            Err(error) => {
                warn!(error = %error, "backend observation failed");
                status.lifecycle = LifecycleState::Error;
                status.tunnel = TunnelState::Down;
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
            return LifecycleDecision::Failed { status, error };
        }

        if !matches!(status.lifecycle, LifecycleState::Ready) {
            status.tunnel = TunnelState::Down;
            self.publish(status.clone());
            return LifecycleDecision::Warming {
                status,
                retry_after_secs: self.timing.retry_after_secs,
            };
        }

        status.tunnel = match self.tunnel.ensure_tunnel().await {
            Ok(tunnel_state) => {
                debug!(tunnel = ?tunnel_state, "tunnel check completed");
                tunnel_state
            }
            Err(error) => {
                warn!(error = %error, "tunnel ensure failed");
                status.lifecycle = LifecycleState::Error;
                status.tunnel = TunnelState::Down;
                return LifecycleDecision::Failed { status, error };
            }
        };

        if !matches!(status.tunnel, TunnelState::Ready) {
            status.lifecycle = LifecycleState::Warming;
        }

        self.publish(status.clone());

        if matches!(status.lifecycle, LifecycleState::Ready)
            && matches!(status.tunnel, TunnelState::Ready)
        {
            LifecycleDecision::Ready(status)
        } else {
            LifecycleDecision::Warming {
                status,
                retry_after_secs: self.timing.retry_after_secs,
            }
        }
    }

    fn bootstrap_deadline(&self) -> Instant {
        let now = self.clock.now().unix_seconds();
        let wake_started_at = self
            .status_tx
            .borrow()
            .last_wake_attempt_at
            .map(Timestamp::unix_seconds)
            .unwrap_or(now);
        let hard_deadline_at = wake_started_at.saturating_add(self.timing.hard_boot_deadline_secs);

        Instant::now() + Duration::from_secs(hard_deadline_at.saturating_sub(now))
    }

    async fn start_bootstrap(self: Arc<Self>) {
        let deadline = self.bootstrap_deadline();
        let poll_interval = Duration::from_millis(self.timing.bootstrap_poll_interval_millis);

        loop {
            if Instant::now() >= deadline {
                let error =
                    LifecycleError::new("backend did not become ready before the boot deadline");
                let mut status = self.status_tx.borrow().clone();
                status.lifecycle = LifecycleState::Error;
                status.tunnel = TunnelState::Down;
                self.finish_bootstrap(status, Some(error)).await;
                return;
            }

            if !poll_interval.is_zero() {
                tokio::time::sleep(poll_interval).await;
            }

            if Instant::now() >= deadline {
                let error =
                    LifecycleError::new("backend did not become ready before the boot deadline");
                let mut status = self.status_tx.borrow().clone();
                status.lifecycle = LifecycleState::Error;
                status.tunnel = TunnelState::Down;
                self.finish_bootstrap(status, Some(error)).await;
                return;
            }

            let status = self.status_tx.borrow().clone();

            let observe_request = self.bootstrap_observe_request().await;

            match self.observe_once(None, observe_request, status).await {
                LifecycleDecision::Ready(status) => {
                    self.finish_bootstrap(status, None).await;
                    return;
                }
                LifecycleDecision::Failed { status, error } => {
                    self.finish_bootstrap(status, Some(error)).await;
                    return;
                }
                LifecycleDecision::Warming { .. } => {}
            }
        }
    }

    async fn finish_bootstrap(&self, status: BackendStatus, error: Option<LifecycleError>) {
        let mut bootstrap = self.bootstrap.lock().await;
        bootstrap.last_error = error.clone();
        self.publish(status);
        bootstrap.active = false;
    }

    async fn wait_for_ready_or_timeout(
        &self,
        mut status_rx: watch::Receiver<BackendStatus>,
    ) -> LifecycleDecision {
        let wait_deadline = Instant::now() + Duration::from_secs(self.timing.cold_wait_budget_secs);

        loop {
            let status = status_rx.borrow().clone();
            let (bootstrap_active, last_error) = {
                let bootstrap = self.bootstrap.lock().await;
                (bootstrap.active, bootstrap.last_error.clone())
            };
            let stale_bootstrap_view =
                bootstrap_active && !matches!(status.lifecycle, LifecycleState::Warming);

            if !stale_bootstrap_view && backend_is_ready(&status) {
                return LifecycleDecision::Ready(status);
            }

            if !stale_bootstrap_view && matches!(status.lifecycle, LifecycleState::Error) {
                let error = last_error
                    .unwrap_or_else(|| LifecycleError::new("backend reported error state"));
                return LifecycleDecision::Failed { status, error };
            }

            let now = Instant::now();
            if now >= wait_deadline {
                return LifecycleDecision::Warming {
                    status: bootstrap_wait_status(status, stale_bootstrap_view),
                    retry_after_secs: self.timing.retry_after_secs,
                };
            }

            let remaining = wait_deadline.saturating_duration_since(now);
            match tokio::time::timeout(remaining, status_rx.changed()).await {
                Ok(Ok(())) => {}
                Ok(Err(_)) | Err(_) => {
                    return LifecycleDecision::Warming {
                        status: bootstrap_wait_status(
                            self.status_tx.borrow().clone(),
                            stale_bootstrap_view,
                        ),
                        retry_after_secs: self.timing.retry_after_secs,
                    };
                }
            }
        }
    }
}

impl<W, S, H, T, C> LifecycleOrchestrator for LifecycleManager<W, S, H, T, C>
where
    W: WakeRequester + Send + Sync + 'static,
    S: SshReadinessProbe + Send + Sync + 'static,
    H: HelperRpc + Send + Sync + 'static,
    T: TunnelOwner + Send + Sync + 'static,
    C: Clock + Send + Sync + 'static,
{
    fn ensure_backend(&self, request: LifecycleRequest) -> LifecycleFuture<'_, LifecycleDecision> {
        Box::pin(async move {
            let initial_status = self.core.status_tx.borrow().clone();
            let bootstrap_active = self.core.bootstrap.lock().await.active;

            if !bootstrap_active && backend_is_ready(&initial_status) {
                return LifecycleDecision::Ready(initial_status);
            }

            let (should_start, status_rx, initial_status) = self
                .core
                .start_or_join_bootstrap(initial_status, &request)
                .await;

            if should_start {
                let observe_request = self.core.bootstrap_observe_request().await;
                match self
                    .core
                    .observe_once(Some(&request), observe_request, initial_status)
                    .await
                {
                    ready @ LifecycleDecision::Ready(_) => {
                        let status = self.core.status_tx.borrow().clone();
                        self.core.finish_bootstrap(status, None).await;
                        return ready;
                    }
                    LifecycleDecision::Failed { status, error } => {
                        self.core
                            .finish_bootstrap(status.clone(), Some(error.clone()))
                            .await;
                        return LifecycleDecision::Failed { status, error };
                    }
                    LifecycleDecision::Warming { .. } => {
                        let core = self.core.clone();
                        tokio::spawn(async move {
                            core.start_bootstrap().await;
                        });
                    }
                }
            }

            self.core.wait_for_ready_or_timeout(status_rx).await
        })
    }

    fn status(&self) -> BackendStatus {
        self.core.status_tx.borrow().clone()
    }

    fn degrade_embeddings(&self, reason: String) -> LifecycleFuture<'_, ()> {
        Box::pin(async move {
            let mut status = self.core.status_tx.borrow().clone();
            status.embeddings = CapabilityState::Degraded;
            status.embeddings_reason = Some(reason);
            self.core.publish(status);
        })
    }

    fn mark_warming(&self) {
        let mut status = self.core.status_tx.borrow().clone();
        status.lifecycle = LifecycleState::Warming;
        status.tunnel = TunnelState::Down;
        self.core.publish(status);
    }
}

fn should_request_wake(lifecycle: LifecycleState) -> bool {
    matches!(
        lifecycle,
        LifecycleState::Cold | LifecycleState::Ready | LifecycleState::Error
    )
}

fn backend_is_ready(status: &BackendStatus) -> bool {
    matches!(status.lifecycle, LifecycleState::Ready) && matches!(status.tunnel, TunnelState::Ready)
}

fn request_priority(request: &LifecycleRequest) -> u8 {
    match request {
        LifecycleRequest::Chat => 0,
        LifecycleRequest::Embeddings => 1,
    }
}

fn bootstrap_wait_status(mut status: BackendStatus, stale_bootstrap_view: bool) -> BackendStatus {
    if stale_bootstrap_view {
        status.lifecycle = LifecycleState::Warming;
        status.tunnel = TunnelState::Down;
    }

    status
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    };

    use tokio::time::sleep;

    use super::*;

    const FAKE_NOW: u64 = 1_717_156_800;

    fn warming_backend() -> ObservedBackendState {
        ObservedBackendState {
            lifecycle: LifecycleState::Warming,
            chat: CapabilityState::Ready,
            embeddings: CapabilityState::Ready,
            embeddings_reason: None,
            error: None,
            llama_server_unit: UnitState::Activating,
            inhibit_unit: UnitState::Activating,
        }
    }

    fn fast_warming_timing() -> LifecycleTiming {
        LifecycleTiming {
            cold_wait_budget_secs: 0,
            hard_boot_deadline_secs: 1,
            bootstrap_poll_interval_millis: 10,
            retry_after_secs: 10,
        }
    }

    #[derive(Clone, Default)]
    struct FakeWakeRequester {
        calls: Arc<Mutex<Vec<LifecycleRequest>>>,
        error: Option<LifecycleError>,
    }

    impl FakeWakeRequester {
        fn with_error(message: &str) -> Self {
            Self {
                calls: Arc::new(Mutex::new(Vec::new())),
                error: Some(LifecycleError::new(message)),
            }
        }

        fn call_count(&self) -> usize {
            self.calls.lock().unwrap().len()
        }

        fn requests(&self) -> Vec<LifecycleRequest> {
            self.calls.lock().unwrap().clone()
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

    #[derive(Clone)]
    struct FakeSshReadinessProbe {
        ready: Arc<AtomicBool>,
    }

    impl FakeSshReadinessProbe {
        fn new(ready: bool) -> Self {
            Self {
                ready: Arc::new(AtomicBool::new(ready)),
            }
        }

        fn set_ready(&self, ready: bool) {
            self.ready.store(ready, Ordering::Relaxed);
        }
    }

    impl SshReadinessProbe for FakeSshReadinessProbe {
        fn is_ready(&self) -> LifecycleFuture<'_, Result<bool, LifecycleError>> {
            let ready = self.ready.load(Ordering::Relaxed);
            Box::pin(async move { Ok(ready) })
        }
    }

    #[derive(Clone)]
    struct FakeHelperRpc {
        calls: Arc<Mutex<Vec<LifecycleRequest>>>,
        observed: Arc<Mutex<ObservedBackendState>>,
    }

    impl Default for FakeHelperRpc {
        fn default() -> Self {
            Self {
                calls: Arc::new(Mutex::new(Vec::new())),
                observed: Arc::new(Mutex::new(warming_backend())),
            }
        }
    }

    impl FakeHelperRpc {
        fn call_count(&self) -> usize {
            self.calls.lock().unwrap().len()
        }

        fn requests(&self) -> Vec<LifecycleRequest> {
            self.calls.lock().unwrap().clone()
        }

        fn ready() -> Self {
            Self {
                calls: Arc::new(Mutex::new(Vec::new())),
                observed: Arc::new(Mutex::new(ObservedBackendState {
                    lifecycle: LifecycleState::Ready,
                    chat: CapabilityState::Ready,
                    embeddings: CapabilityState::Ready,
                    embeddings_reason: None,
                    error: None,
                    llama_server_unit: UnitState::Active,
                    inhibit_unit: UnitState::Active,
                })),
            }
        }

        fn error(message: &str) -> Self {
            Self {
                calls: Arc::new(Mutex::new(Vec::new())),
                observed: Arc::new(Mutex::new(ObservedBackendState {
                    lifecycle: LifecycleState::Error,
                    chat: CapabilityState::Ready,
                    embeddings: CapabilityState::Ready,
                    embeddings_reason: None,
                    error: Some(message.to_string()),
                    llama_server_unit: UnitState::Failed,
                    inhibit_unit: UnitState::Unknown,
                })),
            }
        }

        fn set_observed(&self, observed: ObservedBackendState) {
            *self.observed.lock().unwrap() = observed;
        }
    }

    impl HelperRpc for FakeHelperRpc {
        fn observe_backend(
            &self,
            request: &LifecycleRequest,
        ) -> LifecycleFuture<'_, Result<ObservedBackendState, LifecycleError>> {
            self.calls.lock().unwrap().push(request.clone());
            let observed = self.observed.lock().unwrap().clone();
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
            self.now
        }
    }

    #[tokio::test]
    async fn cold_request_marks_backend_warming_using_only_fakes() {
        let wake = FakeWakeRequester::default();
        let helper = FakeHelperRpc::default();
        let lifecycle = LifecycleManager::new_with_timing(
            wake,
            FakeSshReadinessProbe::new(false),
            helper,
            FakeTunnelOwner::new(TunnelState::Down),
            FakeClock {
                now: Timestamp::new(FAKE_NOW),
            },
            BackendStatus::default(),
            fast_warming_timing(),
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

        let latest = lifecycle.status();
        assert_eq!(latest.lifecycle, LifecycleState::Warming);
        assert_eq!(latest.last_wake_attempt_at, Some(Timestamp::new(FAKE_NOW)));
    }

    #[tokio::test]
    async fn ready_backend_can_be_reached_with_fake_ports_only() {
        let lifecycle = LifecycleManager::new(
            FakeWakeRequester::default(),
            FakeSshReadinessProbe::new(true),
            FakeHelperRpc::ready(),
            FakeTunnelOwner::new(TunnelState::Ready),
            FakeClock {
                now: Timestamp::new(FAKE_NOW),
            },
            BackendStatus::default(),
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

        let latest = lifecycle.status();
        assert_eq!(latest.lifecycle, LifecycleState::Ready);
        assert_eq!(latest.tunnel, TunnelState::Ready);
    }

    #[tokio::test]
    async fn wake_failures_publish_error_without_touching_real_dependencies() {
        let lifecycle = LifecycleManager::new(
            FakeWakeRequester::with_error("wake failed"),
            FakeSshReadinessProbe::new(false),
            FakeHelperRpc::default(),
            FakeTunnelOwner::new(TunnelState::Down),
            FakeClock {
                now: Timestamp::new(FAKE_NOW),
            },
            BackendStatus::default(),
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

        let latest = lifecycle.status();
        assert_eq!(latest.lifecycle, LifecycleState::Error);
        assert_eq!(latest.last_wake_attempt_at, Some(Timestamp::new(FAKE_NOW)));
    }

    #[tokio::test]
    async fn concurrent_cold_requests_share_a_single_bootstrap_and_return_ready() {
        let wake = FakeWakeRequester::default();
        let ssh = FakeSshReadinessProbe::new(false);
        let helper = FakeHelperRpc::default();
        let lifecycle = Arc::new(LifecycleManager::new_with_timing(
            wake.clone(),
            ssh.clone(),
            helper.clone(),
            FakeTunnelOwner::new(TunnelState::Ready),
            FakeClock {
                now: Timestamp::new(FAKE_NOW),
            },
            BackendStatus::default(),
            LifecycleTiming {
                cold_wait_budget_secs: 1,
                hard_boot_deadline_secs: 1,
                bootstrap_poll_interval_millis: 10,
                retry_after_secs: 10,
            },
        ));

        let toggler = tokio::spawn({
            let ssh = ssh.clone();
            let helper = helper.clone();
            async move {
                sleep(Duration::from_millis(20)).await;
                ssh.set_ready(true);
                helper.set_observed(ObservedBackendState {
                    lifecycle: LifecycleState::Ready,
                    chat: CapabilityState::Ready,
                    embeddings: CapabilityState::Ready,
                    embeddings_reason: None,
                    error: None,
                    llama_server_unit: UnitState::Active,
                    inhibit_unit: UnitState::Active,
                });
            }
        });

        let (first, second, _) = tokio::join!(
            lifecycle.ensure_backend(LifecycleRequest::Chat),
            lifecycle.ensure_backend(LifecycleRequest::Embeddings),
            toggler
        );

        assert!(matches!(first, LifecycleDecision::Ready(_)));
        assert!(matches!(second, LifecycleDecision::Ready(_)));
        assert_eq!(wake.call_count(), 1);
        assert_eq!(wake.requests(), vec![LifecycleRequest::Chat]);
        assert_eq!(helper.call_count(), 1);
        assert_eq!(helper.requests(), vec![LifecycleRequest::Embeddings]);
        assert_eq!(lifecycle.status().tunnel, TunnelState::Ready);
    }

    #[tokio::test]
    async fn embeddings_led_bootstrap_observes_embeddings_request_kind() {
        let wake = FakeWakeRequester::default();
        let ssh = FakeSshReadinessProbe::new(false);
        let helper = FakeHelperRpc::default();
        let lifecycle = Arc::new(LifecycleManager::new_with_timing(
            wake.clone(),
            ssh.clone(),
            helper.clone(),
            FakeTunnelOwner::new(TunnelState::Ready),
            FakeClock {
                now: Timestamp::new(FAKE_NOW),
            },
            BackendStatus::default(),
            LifecycleTiming {
                cold_wait_budget_secs: 1,
                hard_boot_deadline_secs: 1,
                bootstrap_poll_interval_millis: 10,
                retry_after_secs: 10,
            },
        ));

        let toggler = tokio::spawn({
            let ssh = ssh.clone();
            let helper = helper.clone();
            async move {
                sleep(Duration::from_millis(20)).await;
                ssh.set_ready(true);
                helper.set_observed(ObservedBackendState {
                    lifecycle: LifecycleState::Ready,
                    chat: CapabilityState::Ready,
                    embeddings: CapabilityState::Ready,
                    embeddings_reason: None,
                    error: None,
                    llama_server_unit: UnitState::Active,
                    inhibit_unit: UnitState::Active,
                });
            }
        });

        let (first, second, _) = tokio::join!(
            lifecycle.ensure_backend(LifecycleRequest::Embeddings),
            lifecycle.ensure_backend(LifecycleRequest::Chat),
            toggler
        );

        assert!(matches!(first, LifecycleDecision::Ready(_)));
        assert!(matches!(second, LifecycleDecision::Ready(_)));
        assert_eq!(wake.requests(), vec![LifecycleRequest::Embeddings]);
        assert_eq!(helper.requests(), vec![LifecycleRequest::Embeddings]);
    }

    #[tokio::test]
    async fn helper_reported_error_returns_failed_decision() {
        let lifecycle = LifecycleManager::new(
            FakeWakeRequester::default(),
            FakeSshReadinessProbe::new(true),
            FakeHelperRpc::error("helper reported backend failure"),
            FakeTunnelOwner::new(TunnelState::Ready),
            FakeClock {
                now: Timestamp::new(FAKE_NOW),
            },
            BackendStatus::default(),
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

        let latest = lifecycle.status();
        assert_eq!(latest.lifecycle, LifecycleState::Error);
        assert_eq!(latest.tunnel, TunnelState::Down);
    }

    #[tokio::test]
    async fn ssh_not_ready_republishes_warming_state_from_ready_snapshot() {
        let wake = FakeWakeRequester::default();
        let initial_status = BackendStatus {
            lifecycle: LifecycleState::Ready,
            tunnel: TunnelState::Ready,
            ..BackendStatus::default()
        };
        let lifecycle = LifecycleManager::new_with_timing(
            wake.clone(),
            FakeSshReadinessProbe::new(false),
            FakeHelperRpc::default(),
            FakeTunnelOwner::new(TunnelState::Ready),
            FakeClock {
                now: Timestamp::new(FAKE_NOW),
            },
            initial_status,
            fast_warming_timing(),
        );

        let decision = lifecycle.ensure_backend(LifecycleRequest::Chat).await;

        match decision {
            LifecycleDecision::Ready(status) => {
                assert_eq!(status.lifecycle, LifecycleState::Ready);
                assert_eq!(status.tunnel, TunnelState::Ready);
            }
            other => panic!("expected ready decision, got {other:?}"),
        }

        let latest = lifecycle.status();
        assert_eq!(latest.lifecycle, LifecycleState::Ready);
        assert_eq!(latest.tunnel, TunnelState::Ready);
        assert_eq!(wake.call_count(), 0);
    }

    #[tokio::test]
    async fn error_snapshot_retries_wake_when_ssh_is_not_ready() {
        let wake = FakeWakeRequester::default();
        let initial_status = BackendStatus {
            lifecycle: LifecycleState::Error,
            tunnel: TunnelState::Down,
            ..BackendStatus::default()
        };
        let lifecycle = LifecycleManager::new_with_timing(
            wake.clone(),
            FakeSshReadinessProbe::new(false),
            FakeHelperRpc::default(),
            FakeTunnelOwner::new(TunnelState::Down),
            FakeClock {
                now: Timestamp::new(FAKE_NOW),
            },
            initial_status,
            fast_warming_timing(),
        );

        let decision = lifecycle.ensure_backend(LifecycleRequest::Chat).await;

        match decision {
            LifecycleDecision::Warming { status, .. } => {
                assert_eq!(status.lifecycle, LifecycleState::Warming);
                assert_eq!(status.tunnel, TunnelState::Down);
            }
            other => panic!("expected warming decision, got {other:?}"),
        }

        assert_eq!(lifecycle.status().lifecycle, LifecycleState::Warming);
        assert_eq!(wake.call_count(), 1);
    }

    #[tokio::test]
    async fn warm_backend_reuses_existing_readiness_without_requesting_wake_or_observe() {
        let wake = FakeWakeRequester::default();
        let helper = FakeHelperRpc::ready();
        let tunnel = FakeTunnelOwner::new(TunnelState::Ready);
        let initial_status = BackendStatus {
            lifecycle: LifecycleState::Ready,
            tunnel: TunnelState::Ready,
            ..BackendStatus::default()
        };
        let lifecycle = LifecycleManager::new(
            wake.clone(),
            FakeSshReadinessProbe::new(true),
            helper.clone(),
            tunnel.clone(),
            FakeClock {
                now: Timestamp::new(FAKE_NOW),
            },
            initial_status,
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
        assert_eq!(helper.call_count(), 0);
        assert_eq!(tunnel.call_count(), 0);
        assert_eq!(lifecycle.status().lifecycle, LifecycleState::Ready);
    }

    #[tokio::test]
    async fn concurrent_warm_requests_do_not_join_bootstrap_wait_path() {
        let wake = FakeWakeRequester::default();
        let helper = FakeHelperRpc::ready();
        let tunnel = FakeTunnelOwner::new(TunnelState::Ready);
        let lifecycle = Arc::new(LifecycleManager::new(
            wake.clone(),
            FakeSshReadinessProbe::new(true),
            helper.clone(),
            tunnel.clone(),
            FakeClock {
                now: Timestamp::new(FAKE_NOW),
            },
            BackendStatus {
                lifecycle: LifecycleState::Ready,
                tunnel: TunnelState::Ready,
                ..BackendStatus::default()
            },
        ));

        let (first, second) = tokio::join!(
            lifecycle.ensure_backend(LifecycleRequest::Chat),
            lifecycle.ensure_backend(LifecycleRequest::Embeddings)
        );

        assert!(matches!(first, LifecycleDecision::Ready(_)));
        assert!(matches!(second, LifecycleDecision::Ready(_)));
        assert_eq!(wake.call_count(), 0);
        assert_eq!(helper.call_count(), 0);
        assert_eq!(tunnel.call_count(), 0);
    }

    #[tokio::test]
    async fn restart_rediscovery_adopts_ready_backend_and_creates_fresh_tunnel() {
        let wake = FakeWakeRequester::default();
        let helper = FakeHelperRpc::ready();
        let tunnel = FakeTunnelOwner::new(TunnelState::Ready);
        let lifecycle = LifecycleManager::new(
            wake.clone(),
            FakeSshReadinessProbe::new(true),
            helper,
            tunnel.clone(),
            FakeClock {
                now: Timestamp::new(FAKE_NOW),
            },
            BackendStatus::default(),
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
        assert_eq!(lifecycle.status().lifecycle, LifecycleState::Ready);
    }

    #[tokio::test]
    async fn request_times_out_but_background_bootstrap_keeps_running() {
        let wake = FakeWakeRequester::default();
        let ssh = FakeSshReadinessProbe::new(false);
        let helper = FakeHelperRpc::default();
        let lifecycle = LifecycleManager::new_with_timing(
            wake.clone(),
            ssh.clone(),
            helper.clone(),
            FakeTunnelOwner::new(TunnelState::Ready),
            FakeClock {
                now: Timestamp::new(FAKE_NOW),
            },
            BackendStatus::default(),
            fast_warming_timing(),
        );

        let decision = lifecycle.ensure_backend(LifecycleRequest::Chat).await;
        assert!(matches!(decision, LifecycleDecision::Warming { .. }));

        ssh.set_ready(true);
        helper.set_observed(ObservedBackendState {
            lifecycle: LifecycleState::Ready,
            chat: CapabilityState::Ready,
            embeddings: CapabilityState::Ready,
            embeddings_reason: None,
            error: None,
            llama_server_unit: UnitState::Active,
            inhibit_unit: UnitState::Active,
        });

        sleep(Duration::from_millis(30)).await;

        let latest = lifecycle.status();
        assert_eq!(latest.lifecycle, LifecycleState::Ready);
        assert_eq!(latest.tunnel, TunnelState::Ready);
        assert_eq!(wake.call_count(), 1);
    }

    #[tokio::test]
    async fn background_bootstrap_fails_at_hard_deadline_after_request_timeout() {
        let lifecycle = LifecycleManager::new_with_timing(
            FakeWakeRequester::default(),
            FakeSshReadinessProbe::new(false),
            FakeHelperRpc::default(),
            FakeTunnelOwner::new(TunnelState::Ready),
            FakeClock {
                now: Timestamp::new(FAKE_NOW),
            },
            BackendStatus::default(),
            LifecycleTiming {
                cold_wait_budget_secs: 0,
                hard_boot_deadline_secs: 0,
                bootstrap_poll_interval_millis: 1,
                retry_after_secs: 10,
            },
        );

        let decision = lifecycle.ensure_backend(LifecycleRequest::Chat).await;
        assert!(matches!(decision, LifecycleDecision::Warming { .. }));

        sleep(Duration::from_millis(20)).await;

        let latest = lifecycle.status();
        assert_eq!(latest.lifecycle, LifecycleState::Error);
        assert_eq!(latest.tunnel, TunnelState::Down);
    }

    #[tokio::test]
    async fn joiner_does_not_observe_stale_ready_during_retry_bootstrap() {
        let wake = FakeWakeRequester::default();
        let ssh = FakeSshReadinessProbe::new(false);
        let lifecycle = Arc::new(LifecycleManager::new_with_timing(
            wake.clone(),
            ssh,
            FakeHelperRpc::default(),
            FakeTunnelOwner::new(TunnelState::Down),
            FakeClock {
                now: Timestamp::new(FAKE_NOW),
            },
            BackendStatus {
                lifecycle: LifecycleState::Ready,
                tunnel: TunnelState::Ready,
                ..BackendStatus::default()
            },
            fast_warming_timing(),
        ));

        let (first, second) = tokio::join!(
            lifecycle.ensure_backend(LifecycleRequest::Chat),
            lifecycle.ensure_backend(LifecycleRequest::Embeddings)
        );

        assert!(matches!(first, LifecycleDecision::Ready { .. }));
        assert!(matches!(second, LifecycleDecision::Ready { .. }));
        assert_eq!(wake.call_count(), 0);
    }

    #[tokio::test]
    async fn joiner_does_not_observe_stale_error_during_retry_bootstrap() {
        let wake = FakeWakeRequester::default();
        let ssh = FakeSshReadinessProbe::new(false);
        let lifecycle = Arc::new(LifecycleManager::new_with_timing(
            wake.clone(),
            ssh,
            FakeHelperRpc::default(),
            FakeTunnelOwner::new(TunnelState::Down),
            FakeClock {
                now: Timestamp::new(FAKE_NOW),
            },
            BackendStatus {
                lifecycle: LifecycleState::Error,
                tunnel: TunnelState::Down,
                ..BackendStatus::default()
            },
            fast_warming_timing(),
        ));

        let (first, second) = tokio::join!(
            lifecycle.ensure_backend(LifecycleRequest::Chat),
            lifecycle.ensure_backend(LifecycleRequest::Embeddings)
        );

        assert!(matches!(first, LifecycleDecision::Warming { .. }));
        assert!(matches!(second, LifecycleDecision::Warming { .. }));
        assert_eq!(wake.call_count(), 1);
    }

    #[tokio::test]
    async fn concurrent_waiters_receive_specific_boot_deadline_failure_reason() {
        let lifecycle = Arc::new(LifecycleManager::new_with_timing(
            FakeWakeRequester::default(),
            FakeSshReadinessProbe::new(false),
            FakeHelperRpc::default(),
            FakeTunnelOwner::new(TunnelState::Ready),
            FakeClock {
                now: Timestamp::new(FAKE_NOW),
            },
            BackendStatus::default(),
            LifecycleTiming {
                cold_wait_budget_secs: 1,
                hard_boot_deadline_secs: 0,
                bootstrap_poll_interval_millis: 1,
                retry_after_secs: 10,
            },
        ));

        let (first, second) = tokio::join!(
            lifecycle.ensure_backend(LifecycleRequest::Chat),
            lifecycle.ensure_backend(LifecycleRequest::Embeddings)
        );

        let expected = "backend did not become ready before the boot deadline";

        match first {
            LifecycleDecision::Failed { error, .. } => assert_eq!(error.message, expected),
            other => panic!("expected failed decision, got {other:?}"),
        }

        match second {
            LifecycleDecision::Failed { error, .. } => assert_eq!(error.message, expected),
            other => panic!("expected failed decision, got {other:?}"),
        }
    }
}

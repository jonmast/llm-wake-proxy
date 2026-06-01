use std::{
    sync::{Arc, RwLock},
    time::{SystemTime, UNIX_EPOCH},
};

use crate::{
    config::AppConfig,
    host::{SshHelperRpc, SshTcpProbe, WolWakeRequester},
    lifecycle::{
        BackendStatePublisher, BackendStatus, Clock, HelperRpc, LifecycleError, LifecycleFuture,
        LifecycleManager, LifecycleOrchestrator, LifecycleRequest, LifecycleState,
        ObservedBackendState, SshReadinessProbe, Timestamp, TunnelOwner, TunnelState,
        WakeRequester,
    },
    scheduler::WarmExecutionScheduler,
};

#[derive(Clone)]
pub struct AppState {
    pub config: AppConfig,
    pub backend: Arc<RwLock<BackendStatus>>,
    pub lifecycle: Arc<dyn LifecycleOrchestrator>,
    pub scheduler: WarmExecutionScheduler,
}

impl AppState {
    pub fn new(config: AppConfig) -> Self {
        let backend = Arc::new(RwLock::new(BackendStatus::default()));
        let lifecycle = Arc::new(LifecycleManager::new(
            NoopWakeRequester,
            NoopSshReadinessProbe,
            NoopHelperRpc,
            NoopTunnelOwner,
            SystemClock,
            SharedBackendState::new(backend.clone()),
        ));
        let scheduler = WarmExecutionScheduler::new(config.warm_execution.clone());

        Self {
            config,
            backend,
            lifecycle,
            scheduler,
        }
    }

    pub fn production(config: AppConfig) -> Self {
        let backend = Arc::new(RwLock::new(BackendStatus::default()));

        let wake = WolWakeRequester::new(
            config.host.wol_mac,
            config.host.wol_broadcast.clone(),
            config.host.wol_port,
        );
        let ssh = SshTcpProbe::new(config.host.host.clone(), config.host.ssh_port);
        let helper = SshHelperRpc::new(
            config.host.ssh_user.clone(),
            config.host.host.clone(),
            config.host.helper_path.clone(),
            config.host.model_path.clone(),
            config.model.alias.clone(),
        );
        let tunnel = NoopTunnelOwner;

        let lifecycle = Arc::new(LifecycleManager::new(
            wake,
            ssh,
            helper,
            tunnel,
            SystemClock,
            SharedBackendState::new(backend.clone()),
        ));
        let scheduler = WarmExecutionScheduler::new(config.warm_execution.clone());

        Self {
            config,
            backend,
            lifecycle,
            scheduler,
        }
    }

    #[cfg(test)]
    pub fn with_lifecycle(
        config: AppConfig,
        backend: Arc<RwLock<BackendStatus>>,
        lifecycle: Arc<dyn LifecycleOrchestrator>,
    ) -> Self {
        let scheduler = WarmExecutionScheduler::new(config.warm_execution.clone());

        Self::with_services(config, backend, lifecycle, scheduler)
    }

    #[cfg(test)]
    pub fn with_services(
        config: AppConfig,
        backend: Arc<RwLock<BackendStatus>>,
        lifecycle: Arc<dyn LifecycleOrchestrator>,
        scheduler: WarmExecutionScheduler,
    ) -> Self {
        Self {
            config,
            backend,
            lifecycle,
            scheduler,
        }
    }
}

#[derive(Clone)]
struct SharedBackendState {
    backend: Arc<RwLock<BackendStatus>>,
}

impl SharedBackendState {
    fn new(backend: Arc<RwLock<BackendStatus>>) -> Self {
        Self { backend }
    }
}

impl BackendStatePublisher for SharedBackendState {
    fn snapshot(&self) -> BackendStatus {
        self.backend
            .read()
            .expect("backend state lock poisoned")
            .clone()
    }

    fn publish(&self, status: BackendStatus) {
        *self.backend.write().expect("backend state lock poisoned") = status;
    }
}

struct NoopWakeRequester;

impl WakeRequester for NoopWakeRequester {
    fn request_wake(
        &self,
        _request: &LifecycleRequest,
    ) -> LifecycleFuture<'_, Result<(), LifecycleError>> {
        Box::pin(async { Ok(()) })
    }
}

struct NoopSshReadinessProbe;

impl SshReadinessProbe for NoopSshReadinessProbe {
    fn is_ready(&self) -> LifecycleFuture<'_, Result<bool, LifecycleError>> {
        Box::pin(async { Ok(false) })
    }
}

struct NoopHelperRpc;

impl HelperRpc for NoopHelperRpc {
    fn observe_backend(
        &self,
        _request: &LifecycleRequest,
    ) -> LifecycleFuture<'_, Result<ObservedBackendState, LifecycleError>> {
        Box::pin(async {
            Ok(ObservedBackendState {
                lifecycle: LifecycleState::Warming,
                chat: crate::lifecycle::CapabilityState::Ready,
                embeddings: crate::lifecycle::CapabilityState::Ready,
                embeddings_reason: None,
                error: None,
                llama_server_unit: crate::lifecycle::UnitState::Unknown,
                inhibit_unit: crate::lifecycle::UnitState::Unknown,
            })
        })
    }
}

pub(crate) struct NoopTunnelOwner;

impl TunnelOwner for NoopTunnelOwner {
    fn ensure_tunnel(&self) -> LifecycleFuture<'_, Result<TunnelState, LifecycleError>> {
        Box::pin(async { Ok(TunnelState::Down) })
    }
}

struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Timestamp {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after unix epoch")
            .as_secs();
        Timestamp::new(now)
    }
}

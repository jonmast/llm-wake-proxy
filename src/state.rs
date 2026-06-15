use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};

use tokio::sync::Semaphore;

use crate::{
    config::AppConfig,
    host::{SshHelperRpc, SshTcpProbe, SshTunnelManager, WolWakeRequester},
    lifecycle::{
        BackendStatus, Clock, HelperRpc, LifecycleError, LifecycleFuture, LifecycleManager,
        LifecycleOrchestrator, LifecycleRequest, LifecycleState, ObservedBackendState,
        SshReadinessProbe, Timestamp, TunnelOwner, TunnelState, WakeRequester,
    },
    metrics::Metrics,
    scheduler::WarmExecutionScheduler,
};

#[derive(Clone)]
pub struct AppState {
    pub config: AppConfig,
    pub lifecycle: Arc<dyn LifecycleOrchestrator>,
    pub scheduler: WarmExecutionScheduler,
    pub metrics: Arc<Metrics>,
    prev_tunnel_was_ready: Arc<AtomicBool>,
    pub cold_start_semaphore: Arc<Semaphore>,
}

impl AppState {
    pub fn check_tunnel_drop(&self, current_tunnel: TunnelState) -> bool {
        let was_ready = self.prev_tunnel_was_ready.swap(
            matches!(current_tunnel, TunnelState::Ready),
            Ordering::Relaxed,
        );
        was_ready && !matches!(current_tunnel, TunnelState::Ready)
    }

    pub fn is_cold(&self) -> bool {
        matches!(self.lifecycle.status().lifecycle, LifecycleState::Cold)
    }
}

impl AppState {
    pub fn new(config: AppConfig) -> Self {
        let cold_start_max = config.cold_start_max_waiting;
        let lifecycle = Arc::new(LifecycleManager::new(
            NoopWakeRequester,
            NoopSshReadinessProbe,
            NoopHelperRpc,
            NoopTunnelOwner,
            SystemClock,
            BackendStatus::default(),
        ));
        let scheduler = WarmExecutionScheduler::new(config.warm_execution.clone());

        Self {
            config,
            lifecycle,
            scheduler,
            metrics: Arc::new(Metrics::default()),
            prev_tunnel_was_ready: Arc::new(AtomicBool::new(false)),
            cold_start_semaphore: Arc::new(Semaphore::new(cold_start_max)),
        }
    }

    pub fn production(config: AppConfig) -> Self {
        let cold_start_max = config.cold_start_max_waiting;

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
            config.host.ssh_key_path.clone(),
        );
        let tunnel = SshTunnelManager::new(
            config.host.ssh_user.clone(),
            config.host.host.clone(),
            config.host.tunnel_local_port,
            config.host.remote_port,
            config.host.ssh_key_path.clone(),
        );

        let lifecycle = Arc::new(LifecycleManager::new(
            wake,
            ssh,
            helper,
            tunnel,
            SystemClock,
            BackendStatus::default(),
        ));
        let scheduler = WarmExecutionScheduler::new(config.warm_execution.clone());

        Self {
            config,
            lifecycle,
            scheduler,
            metrics: Arc::new(Metrics::default()),
            prev_tunnel_was_ready: Arc::new(AtomicBool::new(false)),
            cold_start_semaphore: Arc::new(Semaphore::new(cold_start_max)),
        }
    }

    #[cfg(test)]
    pub fn with_lifecycle(config: AppConfig, lifecycle: Arc<dyn LifecycleOrchestrator>) -> Self {
        let scheduler = WarmExecutionScheduler::new(config.warm_execution.clone());

        Self::with_services(config, lifecycle, scheduler)
    }

    #[cfg(test)]
    pub fn with_services(
        config: AppConfig,
        lifecycle: Arc<dyn LifecycleOrchestrator>,
        scheduler: WarmExecutionScheduler,
    ) -> Self {
        let cold_start_max = config.cold_start_max_waiting;
        Self {
            config,
            lifecycle,
            scheduler,
            metrics: Arc::new(Metrics::default()),
            prev_tunnel_was_ready: Arc::new(AtomicBool::new(false)),
            cold_start_semaphore: Arc::new(Semaphore::new(cold_start_max)),
        }
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

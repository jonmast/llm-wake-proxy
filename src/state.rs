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
    helper::Target,
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
    pub chat_lifecycle: Arc<dyn LifecycleOrchestrator>,
    pub embeddings_lifecycle: Arc<dyn LifecycleOrchestrator>,
    pub scheduler: WarmExecutionScheduler,
    pub metrics: Arc<Metrics>,
    prev_chat_tunnel_was_ready: Arc<AtomicBool>,
    prev_embeddings_tunnel_was_ready: Arc<AtomicBool>,
    pub cold_start_semaphore: Arc<Semaphore>,
}

impl AppState {
    pub fn lifecycle_for(&self, kind: &LifecycleRequest) -> &Arc<dyn LifecycleOrchestrator> {
        match kind {
            LifecycleRequest::Chat => &self.chat_lifecycle,
            LifecycleRequest::Embeddings => &self.embeddings_lifecycle,
        }
    }

    pub fn model_alias_for(&self, kind: &LifecycleRequest) -> &str {
        match kind {
            LifecycleRequest::Chat => &self.config.model.alias,
            LifecycleRequest::Embeddings => self
                .config
                .embeddings
                .backend
                .as_ref()
                .map(|backend| backend.alias.as_str())
                .unwrap_or(&self.config.model.alias),
        }
    }

    pub fn tunnel_port_for(&self, kind: &LifecycleRequest) -> u16 {
        match kind {
            LifecycleRequest::Chat => self.config.host.tunnel_local_port,
            LifecycleRequest::Embeddings => self
                .config
                .embeddings
                .backend
                .as_ref()
                .map(|backend| backend.tunnel_local_port)
                .unwrap_or(self.config.host.tunnel_local_port),
        }
    }

    pub fn is_cold(&self, kind: &LifecycleRequest) -> bool {
        matches!(
            self.lifecycle_for(kind).status().lifecycle,
            LifecycleState::Cold
        )
    }

    pub fn check_tunnel_drop(&self, kind: &LifecycleRequest, current_tunnel: TunnelState) -> bool {
        let flag = match kind {
            LifecycleRequest::Chat => &self.prev_chat_tunnel_was_ready,
            LifecycleRequest::Embeddings => &self.prev_embeddings_tunnel_was_ready,
        };
        let was_ready = flag.swap(
            matches!(current_tunnel, TunnelState::Ready),
            Ordering::Relaxed,
        );
        was_ready && !matches!(current_tunnel, TunnelState::Ready)
    }
}

impl AppState {
    pub fn new(config: AppConfig) -> Self {
        let cold_start_max = config.cold_start_max_waiting;
        let lifecycle: Arc<dyn LifecycleOrchestrator> = Arc::new(LifecycleManager::new(
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
            chat_lifecycle: lifecycle.clone(),
            embeddings_lifecycle: lifecycle,
            scheduler,
            metrics: Arc::new(Metrics::default()),
            prev_chat_tunnel_was_ready: Arc::new(AtomicBool::new(false)),
            prev_embeddings_tunnel_was_ready: Arc::new(AtomicBool::new(false)),
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

        let chat_helper = SshHelperRpc::new(
            config.host.ssh_user.clone(),
            config.host.host.clone(),
            config.host.helper_path.clone(),
            config.host.model_path.clone(),
            config.model.alias.clone(),
            config.host.ssh_key_path.clone(),
            Target::Chat,
        );
        let chat_tunnel = SshTunnelManager::new(
            config.host.ssh_user.clone(),
            config.host.host.clone(),
            config.host.tunnel_local_port,
            config.host.remote_port,
            config.host.ssh_key_path.clone(),
        );

        let chat_lifecycle: Arc<dyn LifecycleOrchestrator> = Arc::new(LifecycleManager::new(
            wake.clone(),
            ssh.clone(),
            chat_helper,
            chat_tunnel,
            SystemClock,
            BackendStatus::default(),
        ));

        let embeddings_lifecycle: Arc<dyn LifecycleOrchestrator> =
            if let Some(backend) = &config.embeddings.backend {
                let embeddings_helper = SshHelperRpc::new(
                    config.host.ssh_user.clone(),
                    config.host.host.clone(),
                    config.host.helper_path.clone(),
                    backend.model_path.clone(),
                    backend.alias.clone(),
                    config.host.ssh_key_path.clone(),
                    Target::Embeddings,
                );
                let embeddings_tunnel = SshTunnelManager::new(
                    config.host.ssh_user.clone(),
                    config.host.host.clone(),
                    backend.tunnel_local_port,
                    backend.remote_port,
                    config.host.ssh_key_path.clone(),
                );

                Arc::new(LifecycleManager::new(
                    wake,
                    ssh,
                    embeddings_helper,
                    embeddings_tunnel,
                    SystemClock,
                    BackendStatus::default(),
                ))
            } else {
                chat_lifecycle.clone()
            };

        let scheduler = WarmExecutionScheduler::new(config.warm_execution.clone());

        Self {
            config,
            chat_lifecycle,
            embeddings_lifecycle,
            scheduler,
            metrics: Arc::new(Metrics::default()),
            prev_chat_tunnel_was_ready: Arc::new(AtomicBool::new(false)),
            prev_embeddings_tunnel_was_ready: Arc::new(AtomicBool::new(false)),
            cold_start_semaphore: Arc::new(Semaphore::new(cold_start_max)),
        }
    }

    #[cfg(test)]
    pub fn with_lifecycle(config: AppConfig, lifecycle: Arc<dyn LifecycleOrchestrator>) -> Self {
        let scheduler = WarmExecutionScheduler::new(config.warm_execution.clone());

        Self::with_services(config, lifecycle, scheduler)
    }

    #[cfg(test)]
    pub fn with_dual_lifecycles(
        config: AppConfig,
        chat_lifecycle: Arc<dyn LifecycleOrchestrator>,
        embeddings_lifecycle: Arc<dyn LifecycleOrchestrator>,
    ) -> Self {
        let cold_start_max = config.cold_start_max_waiting;
        let scheduler = WarmExecutionScheduler::new(config.warm_execution.clone());
        Self {
            config,
            chat_lifecycle,
            embeddings_lifecycle,
            scheduler,
            metrics: Arc::new(Metrics::default()),
            prev_chat_tunnel_was_ready: Arc::new(AtomicBool::new(false)),
            prev_embeddings_tunnel_was_ready: Arc::new(AtomicBool::new(false)),
            cold_start_semaphore: Arc::new(Semaphore::new(cold_start_max)),
        }
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
            chat_lifecycle: lifecycle.clone(),
            embeddings_lifecycle: lifecycle,
            scheduler,
            metrics: Arc::new(Metrics::default()),
            prev_chat_tunnel_was_ready: Arc::new(AtomicBool::new(false)),
            prev_embeddings_tunnel_was_ready: Arc::new(AtomicBool::new(false)),
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

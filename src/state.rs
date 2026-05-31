use std::sync::Arc;

use tokio::sync::RwLock;

use crate::config::AppConfig;

#[derive(Clone)]
pub struct AppState {
    pub config: AppConfig,
    pub backend: Arc<RwLock<BackendStatus>>,
}

impl AppState {
    pub fn new(config: AppConfig) -> Self {
        Self {
            config,
            backend: Arc::new(RwLock::new(BackendStatus::default())),
        }
    }
}

#[derive(Clone, Debug)]
pub struct BackendStatus {
    pub lifecycle: LifecycleState,
    pub chat: CapabilityState,
    pub embeddings: CapabilityState,
    pub embeddings_reason: Option<String>,
    pub tunnel: TunnelState,
    pub last_wake_attempt_at: Option<String>,
    pub lease_expires_at: Option<String>,
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

#[derive(Clone, Copy, Debug)]
pub enum LifecycleState {
    Cold,
    Warming,
    Ready,
    Error,
}

impl LifecycleState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cold => "cold",
            Self::Warming => "warming",
            Self::Ready => "ready",
            Self::Error => "error",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum CapabilityState {
    Ready,
    Degraded,
}

impl CapabilityState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::Degraded => "degraded",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum TunnelState {
    Down,
    Connecting,
    Ready,
}

impl TunnelState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Down => "down",
            Self::Connecting => "connecting",
            Self::Ready => "ready",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum UnitState {
    Unknown,
    Inactive,
    Activating,
    Active,
    Failed,
}

impl UnitState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::Inactive => "inactive",
            Self::Activating => "activating",
            Self::Active => "active",
            Self::Failed => "failed",
        }
    }
}

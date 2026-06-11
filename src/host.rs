use std::time::Duration;

use crate::helper::{HelperStatus, StatusResponse};
use crate::lifecycle::{
    CapabilityState, HelperRpc, LifecycleError, LifecycleFuture, LifecycleRequest,
    LifecycleState, ObservedBackendState, SshReadinessProbe, TunnelOwner, TunnelState,
    WakeRequester,
};

// ===== Wake-on-LAN =====

#[derive(Clone, Debug)]
pub struct WolWakeRequester {
    mac: [u8; 6],
    broadcast: String,
    port: u16,
}

impl WolWakeRequester {
    pub fn new(mac: [u8; 6], broadcast: String, port: u16) -> Self {
        Self { mac, broadcast, port }
    }
}

impl WakeRequester for WolWakeRequester {
    fn request_wake(
        &self,
        _request: &LifecycleRequest,
    ) -> LifecycleFuture<'_, Result<(), LifecycleError>> {
        let mac = self.mac;
        let broadcast = self.broadcast.clone();
        let port = self.port;
        Box::pin(async move {
            let mut packet = Vec::with_capacity(102);
            packet.extend_from_slice(&[0xFF; 6]);
            for _ in 0..16 {
                packet.extend_from_slice(&mac);
            }

            let socket = tokio::net::UdpSocket::bind("0.0.0.0:0")
                .await
                .map_err(|e| LifecycleError::new(format!("WOL bind failed: {e}")))?;
            socket
                .set_broadcast(true)
                .map_err(|e| LifecycleError::new(format!("WOL broadcast failed: {e}")))?;
            socket
                .send_to(&packet, format!("{broadcast}:{port}"))
                .await
                .map_err(|e| LifecycleError::new(format!("WOL send failed: {e}")))?;
            Ok(())
        })
    }
}

// ===== SSH Readiness Probe =====

#[derive(Clone, Debug)]
pub struct SshTcpProbe {
    host: String,
    port: u16,
}

impl SshTcpProbe {
    pub fn new(host: String, port: u16) -> Self {
        Self { host, port }
    }
}

impl SshReadinessProbe for SshTcpProbe {
    fn is_ready(&self) -> LifecycleFuture<'_, Result<bool, LifecycleError>> {
        let addr = format!("{}:{}", self.host, self.port);
        Box::pin(async move {
            match tokio::time::timeout(
                Duration::from_secs(10),
                tokio::net::TcpStream::connect(&addr),
            )
            .await
            {
                Ok(Ok(_)) => Ok(true),
                _ => Ok(false),
            }
        })
    }
}

// ===== SSH Helper RPC =====

#[derive(Clone, Debug)]
pub struct SshHelperRpc {
    ssh_user: String,
    ssh_host: String,
    helper_path: String,
    model_path: String,
    model_alias: String,
    ssh_key_path: String,
}

impl SshHelperRpc {
    pub fn new(
        ssh_user: String,
        ssh_host: String,
        helper_path: String,
        model_path: String,
        model_alias: String,
        ssh_key_path: String,
    ) -> Self {
        Self {
            ssh_user,
            ssh_host,
            helper_path,
            model_path,
            model_alias,
            ssh_key_path,
        }
    }

    async fn run_helper_cmd(&self, args: &[&str]) -> Result<String, LifecycleError> {
        let mut cmd = tokio::process::Command::new("ssh");
        cmd.args([
            "-o",
            "ConnectTimeout=10",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-i",
            &self.ssh_key_path,
        ]);
        cmd.arg(format!("{}@{}", self.ssh_user, self.ssh_host));
        for arg in args {
            cmd.arg(arg);
        }

        let output = cmd
            .output()
            .await
            .map_err(|e| LifecycleError::new(format!("ssh failed: {e}")))?;

        if output.status.success() {
            String::from_utf8(output.stdout)
                .map_err(|e| LifecycleError::new(format!("invalid UTF-8 from helper: {e}")))
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(LifecycleError::new(format!(
                "helper command failed: {stderr}"
            )))
        }
    }

    async fn ensure_started(&self) -> Result<ObservedBackendState, LifecycleError> {
        let output = self
            .run_helper_cmd(&[
                "env",
                &format!("EXPECTED_MODEL_PATH={}", self.model_path),
                &self.helper_path,
                "ensure-started",
                &self.model_alias,
            ])
            .await?;

        let ensure: crate::helper::EnsureStartedResponse =
            serde_json::from_str(&output).map_err(|e| {
                LifecycleError::new(format!("failed to parse ensure-started response: {e}"))
            })?;

        let lifecycle = match ensure.status {
            HelperStatus::Ok if ensure.model_verified => LifecycleState::Ready,
            HelperStatus::Ok => LifecycleState::Warming,
            _ => LifecycleState::Error,
        };

        let error = match ensure.status {
            HelperStatus::Ok => None,
            _ => Some(ensure.message.clone()),
        };

        let llama_server_unit = map_unit_state(
            ensure
                .llama_server
                .as_ref()
                .map(|u| u.active_state.as_str()),
        );

        Ok(ObservedBackendState {
            lifecycle,
            chat: CapabilityState::Ready,
            embeddings: CapabilityState::Ready,
            embeddings_reason: None,
            error,
            llama_server_unit,
            inhibit_unit: crate::lifecycle::UnitState::Unknown,
        })
    }

    async fn fetch_status(&self) -> Result<ObservedBackendState, LifecycleError> {
        let output = self
            .run_helper_cmd(&[&self.helper_path, "status"])
            .await?;

        let parsed: StatusResponse = serde_json::from_str(&output)
            .map_err(|e| LifecycleError::new(format!("failed to parse status response: {e}")))?;

        let llama_server_unit = map_unit_state(
            parsed
                .llama_server
                .as_ref()
                .map(|u| u.active_state.as_str()),
        );
        let inhibit_unit = map_unit_state(
            parsed
                .inhibit_holder
                .as_ref()
                .map(|u| u.active_state.as_str()),
        );

        let lifecycle = if matches!(parsed.status, HelperStatus::Error) {
            LifecycleState::Error
        } else if llama_server_unit == crate::lifecycle::UnitState::Active {
            LifecycleState::Ready
        } else {
            LifecycleState::Warming
        };

        Ok(ObservedBackendState {
            lifecycle,
            chat: CapabilityState::Ready,
            embeddings: CapabilityState::Ready,
            embeddings_reason: None,
            error: None,
            llama_server_unit,
            inhibit_unit,
        })
    }
}

impl HelperRpc for SshHelperRpc {
    fn observe_backend(
        &self,
        _request: &LifecycleRequest,
    ) -> LifecycleFuture<'_, Result<ObservedBackendState, LifecycleError>> {
        Box::pin(async move {
            let mut state = self.ensure_started().await?;

            if matches!(state.lifecycle, LifecycleState::Ready) {
                if let Ok(status_state) = self.fetch_status().await {
                    state.llama_server_unit = status_state.llama_server_unit;
                    state.inhibit_unit = status_state.inhibit_unit;
                }
            }

            Ok(state)
        })
    }
}

fn map_unit_state(active_state: Option<&str>) -> crate::lifecycle::UnitState {
    match active_state {
        Some("active") => crate::lifecycle::UnitState::Active,
        Some("activating" | "reloading") => crate::lifecycle::UnitState::Activating,
        Some("inactive") => crate::lifecycle::UnitState::Inactive,
        Some("failed" | "error") => crate::lifecycle::UnitState::Failed,
        _ => crate::lifecycle::UnitState::Unknown,
    }
}

// ===== SSH Tunnel Manager =====

use tokio::sync::Mutex;
use tokio::process::Child;

pub struct SshTunnelManager {
    ssh_user: String,
    ssh_host: String,
    local_port: u16,
    remote_port: u16,
    ssh_key_path: String,
    proc: Mutex<Option<Child>>,
}

impl SshTunnelManager {
    pub fn new(ssh_user: String, ssh_host: String, local_port: u16, remote_port: u16, ssh_key_path: String) -> Self {
        Self {
            ssh_user,
            ssh_host,
            local_port,
            remote_port,
            ssh_key_path,
            proc: Mutex::new(None),
        }
    }

    async fn create_tunnel(&self) -> Result<Child, LifecycleError> {
        let mut child = tokio::process::Command::new("ssh")
            .kill_on_drop(true)
            .arg("-o")
            .arg("ConnectTimeout=10")
            .arg("-o")
            .arg("BatchMode=yes")
            .arg("-o")
            .arg("StrictHostKeyChecking=accept-new")
            .arg("-o")
            .arg("ExitOnForwardFailure=yes")
            .arg("-i")
            .arg(&self.ssh_key_path)
            .arg("-L")
            .arg(format!(
                "127.0.0.1:{}:127.0.0.1:{}",
                self.local_port, self.remote_port
            ))
            .arg("-N")
            .arg(format!("{}@{}", self.ssh_user, self.ssh_host))
            .spawn()
            .map_err(|e| LifecycleError::new(format!("failed to spawn SSH tunnel: {e}")))?;

        tokio::time::sleep(Duration::from_millis(500)).await;

        match child.try_wait() {
            Ok(Some(status)) => Err(LifecycleError::new(format!(
                "SSH tunnel exited immediately with status: {status}"
            ))),
            Ok(None) => Ok(child),
            Err(e) => Err(LifecycleError::new(format!(
                "failed to check tunnel: {e}"
            ))),
        }
    }
}

impl TunnelOwner for SshTunnelManager {
    fn ensure_tunnel(&self) -> LifecycleFuture<'_, Result<TunnelState, LifecycleError>> {
        Box::pin(async move {
            let mut proc = self.proc.lock().await;

            if let Some(ref mut child) = *proc {
                match child.try_wait() {
                    Ok(Some(_)) => {
                        *proc = None;
                        match self.create_tunnel().await {
                            Ok(c) => {
                                *proc = Some(c);
                                Ok(TunnelState::Ready)
                            }
                            Err(e) => Err(e),
                        }
                    }
                    Ok(None) => Ok(TunnelState::Ready),
                    Err(_) => Ok(TunnelState::Down),
                }
            } else {
                match self.create_tunnel().await {
                    Ok(child) => {
                        *proc = Some(child);
                        Ok(TunnelState::Ready)
                    }
                    Err(e) => Err(e),
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_unit_state_known_values() {
        assert_eq!(
            map_unit_state(Some("active")),
            crate::lifecycle::UnitState::Active
        );
        assert_eq!(
            map_unit_state(Some("inactive")),
            crate::lifecycle::UnitState::Inactive
        );
        assert_eq!(
            map_unit_state(Some("activating")),
            crate::lifecycle::UnitState::Activating
        );
        assert_eq!(
            map_unit_state(Some("failed")),
            crate::lifecycle::UnitState::Failed
        );
        assert_eq!(
            map_unit_state(None),
            crate::lifecycle::UnitState::Unknown
        );
        assert_eq!(
            map_unit_state(Some("garbage")),
            crate::lifecycle::UnitState::Unknown
        );
    }
}

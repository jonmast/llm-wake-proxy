use std::collections::HashMap;
use std::process::Command;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HelperStatus {
    Ok,
    Mismatch,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    Chat,
    Embeddings,
}

impl Target {
    pub fn as_str(&self) -> &'static str {
        match self {
            Target::Chat => "chat",
            Target::Embeddings => "embeddings",
        }
    }
}

impl std::str::FromStr for Target {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "chat" => Ok(Target::Chat),
            "embeddings" => Ok(Target::Embeddings),
            other => Err(format!(
                "unknown target '{other}', expected 'chat' or 'embeddings'"
            )),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnitInfo {
    pub unit_name: String,
    pub active_state: String,
    pub sub_state: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub model_alias: String,
    pub model_path: String,
    pub serving: bool,
    pub ready: bool,
    pub reported_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusResponse {
    pub status: HelperStatus,
    pub message: String,
    pub llama_server: Option<UnitInfo>,
    pub inhibit_holder: Option<UnitInfo>,
    pub active_model: Option<ModelInfo>,
    #[serde(default)]
    pub llama_server_embeddings: Option<UnitInfo>,
    #[serde(default)]
    pub active_model_embeddings: Option<ModelInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsureStartedResponse {
    pub status: HelperStatus,
    pub message: String,
    pub model_alias: String,
    pub model_path: String,
    pub startup_triggered: bool,
    pub server_ready: bool,
    pub model_verified: bool,
    pub llama_server: Option<UnitInfo>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LeaseAcquireResponse {
    pub status: HelperStatus,
    pub message: String,
    pub inhibit_holder: Option<UnitInfo>,
    pub ttl_secs: u64,
    pub renewed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct LeaseReleaseResponse {
    pub status: HelperStatus,
    pub message: String,
    pub inhibit_holder: Option<UnitInfo>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LeaseInspectResponse {
    pub status: HelperStatus,
    pub message: String,
    pub inhibit_holder: Option<UnitInfo>,
}

pub trait SystemdControl: Send + Sync {
    fn show_unit(&self, unit: &str) -> Result<HashMap<String, String>, String>;
    fn is_active(&self, unit: &str) -> Result<bool, String>;
    fn start_unit(&self, unit: &str) -> Result<(), String>;
    fn stop_unit(&self, unit: &str) -> Result<(), String> {
        let _ = unit;
        Err("stop_unit not implemented".to_string())
    }
    fn replace_transient_unit(
        &self,
        unit: &str,
        description: &str,
        args: &[&str],
    ) -> Result<(), String> {
        let _ = (unit, description, args);
        Err("replace_transient_unit not implemented".to_string())
    }
}

pub trait ServerCheck: Send + Sync {
    fn get_text(&self, url: &str) -> Result<String, String>;
}

pub struct RealSystemd;

impl SystemdControl for RealSystemd {
    fn show_unit(&self, unit: &str) -> Result<HashMap<String, String>, String> {
        let output = Command::new("systemctl")
            .args([
                "--user",
                "show",
                unit,
                "--property=ActiveState",
                "--property=SubState",
                "--property=Description",
            ])
            .output()
            .map_err(|e| format!("failed to run systemctl: {e}"))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("systemctl show failed: {stderr}"));
        }
        parse_systemctl_output(&output.stdout)
    }

    fn is_active(&self, unit: &str) -> Result<bool, String> {
        let output = Command::new("systemctl")
            .args(["--user", "is-active", unit])
            .output()
            .map_err(|e| format!("failed to run systemctl: {e}"))?;
        Ok(output.status.success())
    }

    fn start_unit(&self, unit: &str) -> Result<(), String> {
        let output = Command::new("systemctl")
            .args(["--user", "start", unit])
            .output()
            .map_err(|e| format!("failed to run systemctl: {e}"))?;
        if output.status.success() {
            Ok(())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(format!("systemctl start failed: {stderr}"))
        }
    }

    fn stop_unit(&self, unit: &str) -> Result<(), String> {
        let output = Command::new("systemctl")
            .args(["--user", "stop", unit])
            .output()
            .map_err(|e| format!("failed to run systemctl: {e}"))?;
        if output.status.success() {
            Ok(())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(format!("systemctl stop failed: {stderr}"))
        }
    }

    fn replace_transient_unit(
        &self,
        unit: &str,
        description: &str,
        args: &[&str],
    ) -> Result<(), String> {
        let mut cmd = Command::new("systemd-run");
        cmd.args([
            "--user",
            "--unit",
            unit,
            "--replace",
            "--description",
            description,
            "--collect",
        ]);
        cmd.args(args);
        let output = cmd
            .output()
            .map_err(|e| format!("failed to run systemd-run: {e}"))?;
        if output.status.success() {
            Ok(())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(format!("systemd-run failed: {stderr}"))
        }
    }
}

pub struct CurlHttp;

impl ServerCheck for CurlHttp {
    fn get_text(&self, url: &str) -> Result<String, String> {
        let output = Command::new("curl")
            .args(["-sf", url])
            .output()
            .map_err(|e| format!("failed to run curl: {e}"))?;
        if output.status.success() {
            String::from_utf8(output.stdout).map_err(|e| format!("invalid UTF-8: {e}"))
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(format!("curl failed: {stderr}"))
        }
    }
}

fn parse_systemctl_output(bytes: &[u8]) -> Result<HashMap<String, String>, String> {
    let text = String::from_utf8(bytes.to_vec())
        .map_err(|e| format!("invalid UTF-8 in systemctl output: {e}"))?;
    let mut map = HashMap::new();
    for line in text.lines() {
        if let Some((key, value)) = line.split_once('=') {
            map.insert(key.to_string(), value.to_string());
        }
    }
    Ok(map)
}

fn build_server_url(port: u16, path: &str) -> String {
    format!("http://127.0.0.1:{port}{path}")
}

#[derive(Clone, Debug)]
pub struct HelperConfig {
    pub llama_server_unit: String,
    pub inhibit_holder_unit: String,
    pub llama_server_port: u16,
    pub llama_server_embeddings_unit: String,
    pub llama_server_embeddings_port: u16,
    pub server_start_timeout_secs: u64,
}

impl Default for HelperConfig {
    fn default() -> Self {
        Self {
            llama_server_unit: "llama-server".to_string(),
            inhibit_holder_unit: "llm-wake-proxy-inhibit".to_string(),
            llama_server_port: 8080,
            llama_server_embeddings_unit: "llama-server-embeddings".to_string(),
            llama_server_embeddings_port: 8081,
            server_start_timeout_secs: 60,
        }
    }
}

impl HelperConfig {
    pub fn from_env() -> Self {
        Self {
            llama_server_unit: std::env::var("LLAMA_SERVER_UNIT")
                .unwrap_or_else(|_| "llama-server".to_string()),
            inhibit_holder_unit: std::env::var("INHIBIT_HOLDER_UNIT")
                .unwrap_or_else(|_| "llm-wake-proxy-inhibit".to_string()),
            llama_server_port: std::env::var("LLAMA_SERVER_PORT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8080),
            llama_server_embeddings_unit: std::env::var("LLAMA_SERVER_EMBEDDINGS_UNIT")
                .unwrap_or_else(|_| "llama-server-embeddings".to_string()),
            llama_server_embeddings_port: std::env::var("LLAMA_SERVER_EMBEDDINGS_PORT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8081),
            server_start_timeout_secs: std::env::var("SERVER_START_TIMEOUT_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(60),
        }
    }

    pub fn unit_for(&self, target: Target) -> &str {
        match target {
            Target::Chat => &self.llama_server_unit,
            Target::Embeddings => &self.llama_server_embeddings_unit,
        }
    }

    pub fn port_for(&self, target: Target) -> u16 {
        match target {
            Target::Chat => self.llama_server_port,
            Target::Embeddings => self.llama_server_embeddings_port,
        }
    }
}

pub struct Helper {
    pub config: HelperConfig,
    sys: Box<dyn SystemdControl>,
    http: Box<dyn ServerCheck>,
}

impl Helper {
    pub fn new(config: HelperConfig) -> Self {
        Self {
            config,
            sys: Box::new(RealSystemd),
            http: Box::new(CurlHttp),
        }
    }

    pub fn with_services(
        config: HelperConfig,
        sys: Box<dyn SystemdControl>,
        http: Box<dyn ServerCheck>,
    ) -> Self {
        Self { config, sys, http }
    }

    pub fn status(&self) -> StatusResponse {
        let llama_server = self.collect_unit_info(&self.config.llama_server_unit);
        let inhibit_holder = self.collect_unit_info(&self.config.inhibit_holder_unit);
        let active_model = self.active_model_for(self.config.llama_server_port);

        let llama_server_embeddings =
            self.collect_unit_info(&self.config.llama_server_embeddings_unit);
        let active_model_embeddings =
            self.active_model_for(self.config.llama_server_embeddings_port);

        StatusResponse {
            status: HelperStatus::Ok,
            message: "current host unit state".to_string(),
            llama_server,
            inhibit_holder,
            active_model,
            llama_server_embeddings,
            active_model_embeddings,
        }
    }

    fn active_model_for(&self, port: u16) -> Option<ModelInfo> {
        if self.is_server_alive(port) {
            match self.check_model_identity(port) {
                Ok(reported_id) => Some(ModelInfo {
                    model_alias: reported_id.as_deref().unwrap_or_default().to_string(),
                    model_path: reported_id.unwrap_or_default(),
                    serving: true,
                    ready: true,
                    reported_id: None,
                }),
                Err(_) => Some(ModelInfo {
                    model_alias: String::new(),
                    model_path: String::new(),
                    serving: true,
                    ready: false,
                    reported_id: None,
                }),
            }
        } else {
            None
        }
    }

    pub fn ensure_started(
        &self,
        target: Target,
        model_alias: &str,
        model_path: &str,
    ) -> EnsureStartedResponse {
        let unit = self.config.unit_for(target);
        let port = self.config.port_for(target);
        let mut startup_triggered = false;

        let is_active = match self.sys.is_active(unit) {
            Ok(active) => active,
            Err(e) => {
                eprintln!("Failed to check unit state: {e}");
                false
            }
        };

        if !is_active {
            eprintln!("Starting systemd unit: {unit}");
            match self.sys.start_unit(unit) {
                Ok(()) => startup_triggered = true,
                Err(e) => {
                    return EnsureStartedResponse {
                        status: HelperStatus::Error,
                        message: format!("failed to start llama-server: {e}"),
                        model_alias: model_alias.to_string(),
                        model_path: model_path.to_string(),
                        startup_triggered: false,
                        server_ready: false,
                        model_verified: false,
                        llama_server: Some(UnitInfo {
                            unit_name: unit.to_string(),
                            active_state: "failed".to_string(),
                            sub_state: "startup_error".to_string(),
                            description: e,
                        }),
                    };
                }
            }
        }

        let server_ready = self.wait_for_server(port, self.config.server_start_timeout_secs);

        if !server_ready {
            let unit_info = self.collect_unit_info(unit).unwrap_or(UnitInfo {
                unit_name: unit.to_string(),
                active_state: "unknown".to_string(),
                sub_state: "server_not_responding".to_string(),
                description: String::new(),
            });
            return EnsureStartedResponse {
                status: HelperStatus::Error,
                message: format!(
                    "llama-server did not become ready within {} seconds",
                    self.config.server_start_timeout_secs
                ),
                model_alias: model_alias.to_string(),
                model_path: model_path.to_string(),
                startup_triggered,
                server_ready: false,
                model_verified: false,
                llama_server: Some(unit_info),
            };
        }

        let model_verified = match self.check_model_identity(port) {
            Ok(Some(reported_id)) => {
                if reported_id.contains(model_path) || model_path.contains(&reported_id) {
                    true
                } else {
                    let unit_info = self.collect_unit_info(unit).unwrap_or(UnitInfo {
                        unit_name: unit.to_string(),
                        active_state: "active".to_string(),
                        sub_state: "running".to_string(),
                        description: String::new(),
                    });
                    return EnsureStartedResponse {
                        status: HelperStatus::Mismatch,
                        message: format!(
                            "expected model path '{model_path}', but server reports '{reported_id}'"
                        ),
                        model_alias: model_alias.to_string(),
                        model_path: model_path.to_string(),
                        startup_triggered,
                        server_ready,
                        model_verified: false,
                        llama_server: Some(unit_info),
                    };
                }
            }
            Ok(None) => {
                let unit_info = self.collect_unit_info(unit).unwrap_or(UnitInfo {
                    unit_name: unit.to_string(),
                    active_state: "active".to_string(),
                    sub_state: "running".to_string(),
                    description: String::new(),
                });
                return EnsureStartedResponse {
                    status: HelperStatus::Error,
                    message: "server responded but returned no model identifiers".to_string(),
                    model_alias: model_alias.to_string(),
                    model_path: model_path.to_string(),
                    startup_triggered,
                    server_ready,
                    model_verified: false,
                    llama_server: Some(unit_info),
                };
            }
            Err(e) => {
                let unit_info = self.collect_unit_info(unit).unwrap_or(UnitInfo {
                    unit_name: unit.to_string(),
                    active_state: "unknown".to_string(),
                    sub_state: "model_identity_check_error".to_string(),
                    description: e.clone(),
                });
                return EnsureStartedResponse {
                    status: HelperStatus::Error,
                    message: format!("model identity check failed: {e}"),
                    model_alias: model_alias.to_string(),
                    model_path: model_path.to_string(),
                    startup_triggered,
                    server_ready,
                    model_verified: false,
                    llama_server: Some(unit_info),
                };
            }
        };

        let unit_info = self.collect_unit_info(unit).unwrap_or(UnitInfo {
            unit_name: unit.to_string(),
            active_state: "active".to_string(),
            sub_state: "running".to_string(),
            description: String::new(),
        });

        EnsureStartedResponse {
            status: HelperStatus::Ok,
            message: if startup_triggered {
                "model started and verified successfully".to_string()
            } else {
                "model is already running and ready".to_string()
            },
            model_alias: model_alias.to_string(),
            model_path: model_path.to_string(),
            startup_triggered,
            server_ready,
            model_verified,
            llama_server: Some(unit_info),
        }
    }

    pub fn lease_acquire(&self, ttl_secs: u64) -> LeaseAcquireResponse {
        let ttl_secs = ttl_secs.clamp(60, 86_400);
        let was_active = self
            .sys
            .is_active(&self.config.inhibit_holder_unit)
            .unwrap_or(false);

        match self.sys.replace_transient_unit(
            &self.config.inhibit_holder_unit,
            "llm-wake-proxy inhibit holder",
            &[
                "systemd-inhibit",
                "--what=sleep",
                "--who=llm-wake-proxy",
                "--why=Keep awake for active proxy session",
                "sleep",
                &ttl_secs.to_string(),
            ],
        ) {
            Ok(()) => {
                let unit = self.collect_unit_info(&self.config.inhibit_holder_unit);
                let is_verified_active = unit
                    .as_ref()
                    .map(|u| u.active_state == "active")
                    .unwrap_or(false);

                if !is_verified_active {
                    return LeaseAcquireResponse {
                        status: HelperStatus::Error,
                        message: "lease was created but the inhibit unit is not active".to_string(),
                        inhibit_holder: unit,
                        ttl_secs,
                        renewed: false,
                    };
                }

                LeaseAcquireResponse {
                    status: HelperStatus::Ok,
                    message: if was_active {
                        "lease renewed".to_string()
                    } else {
                        "lease acquired".to_string()
                    },
                    inhibit_holder: unit,
                    ttl_secs,
                    renewed: was_active,
                }
            }
            Err(e) => LeaseAcquireResponse {
                status: HelperStatus::Error,
                message: format!("failed to acquire lease: {e}"),
                inhibit_holder: None,
                ttl_secs,
                renewed: false,
            },
        }
    }

    pub fn lease_release(&self) -> LeaseReleaseResponse {
        match self.sys.stop_unit(&self.config.inhibit_holder_unit) {
            Ok(()) => {
                let unit = self.collect_unit_info(&self.config.inhibit_holder_unit);
                LeaseReleaseResponse {
                    status: HelperStatus::Ok,
                    message: "lease released".to_string(),
                    inhibit_holder: unit,
                }
            }
            Err(e) => LeaseReleaseResponse {
                status: HelperStatus::Error,
                message: format!("failed to release lease: {e}"),
                inhibit_holder: None,
            },
        }
    }

    pub fn lease_inspect(&self) -> LeaseInspectResponse {
        match self.sys.show_unit(&self.config.inhibit_holder_unit) {
            Ok(props) => {
                let active_state = props
                    .get("ActiveState")
                    .cloned()
                    .unwrap_or_else(|| "unknown".to_string());
                let sub_state = props
                    .get("SubState")
                    .cloned()
                    .unwrap_or_else(|| "unknown".to_string());
                let description = props.get("Description").cloned().unwrap_or_default();
                LeaseInspectResponse {
                    status: HelperStatus::Ok,
                    message: "current lease state".to_string(),
                    inhibit_holder: Some(UnitInfo {
                        unit_name: self.config.inhibit_holder_unit.clone(),
                        active_state,
                        sub_state,
                        description,
                    }),
                }
            }
            Err(e) => LeaseInspectResponse {
                status: HelperStatus::Error,
                message: format!("failed to inspect lease: {e}"),
                inhibit_holder: None,
            },
        }
    }

    fn collect_unit_info(&self, unit_name: &str) -> Option<UnitInfo> {
        match self.sys.show_unit(unit_name) {
            Ok(props) => Some(UnitInfo {
                unit_name: unit_name.to_string(),
                active_state: props
                    .get("ActiveState")
                    .cloned()
                    .unwrap_or_else(|| "unknown".to_string()),
                sub_state: props
                    .get("SubState")
                    .cloned()
                    .unwrap_or_else(|| "unknown".to_string()),
                description: props.get("Description").cloned().unwrap_or_default(),
            }),
            Err(e) => Some(UnitInfo {
                unit_name: unit_name.to_string(),
                active_state: "error".to_string(),
                sub_state: e,
                description: String::new(),
            }),
        }
    }

    fn is_server_alive(&self, port: u16) -> bool {
        let url = build_server_url(port, "/v1/models");
        self.http.get_text(&url).is_ok()
    }

    fn wait_for_server(&self, port: u16, timeout_secs: u64) -> bool {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
        while std::time::Instant::now() < deadline {
            if self.is_server_alive(port) {
                return true;
            }
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
        false
    }

    fn check_model_identity(&self, port: u16) -> Result<Option<String>, String> {
        let url = build_server_url(port, "/v1/models");
        let body = self.http.get_text(&url)?;
        let json: serde_json::Value = serde_json::from_str(&body)
            .map_err(|e| format!("failed to parse models response: {e}"))?;

        Ok(json["data"]
            .as_array()
            .and_then(|arr| arr.first())
            .and_then(|first| first["id"].as_str())
            .map(|s| s.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn status_response_serializes_to_json() {
        let response = StatusResponse {
            status: HelperStatus::Ok,
            message: "current host unit state".to_string(),
            llama_server: Some(UnitInfo {
                unit_name: "llama-server".to_string(),
                active_state: "active".to_string(),
                sub_state: "running".to_string(),
                description: "llama.cpp server".to_string(),
            }),
            inhibit_holder: Some(UnitInfo {
                unit_name: "llm-wake-proxy-inhibit".to_string(),
                active_state: "active".to_string(),
                sub_state: "running".to_string(),
                description: "Inhibit holder for llm-wake-proxy".to_string(),
            }),
            active_model: Some(ModelInfo {
                model_alias: "llama-3.2-3b".to_string(),
                model_path: "/models/llama-3.2-3b.Q4_K_M.gguf".to_string(),
                serving: true,
                ready: true,
                reported_id: Some("/models/llama-3.2-3b.Q4_K_M.gguf".to_string()),
            }),
            llama_server_embeddings: None,
            active_model_embeddings: None,
        };

        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["status"], "ok");
        assert_eq!(json["llama_server"]["unit_name"], "llama-server");
        assert_eq!(json["llama_server"]["active_state"], "active");
        assert_eq!(json["active_model"]["model_alias"], "llama-3.2-3b");
        assert_eq!(
            json["active_model"]["reported_id"],
            "/models/llama-3.2-3b.Q4_K_M.gguf"
        );
    }

    #[test]
    fn status_response_handles_missing_units() {
        let response = StatusResponse {
            status: HelperStatus::Ok,
            message: "current host unit state".to_string(),
            llama_server: None,
            inhibit_holder: None,
            active_model: None,
            llama_server_embeddings: None,
            active_model_embeddings: None,
        };

        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["llama_server"], serde_json::Value::Null);
        assert_eq!(json["active_model"], serde_json::Value::Null);
    }

    #[test]
    fn ensure_started_ok_serializes_to_json() {
        let response = EnsureStartedResponse {
            status: HelperStatus::Ok,
            message: "model is already running and ready".to_string(),
            model_alias: "llama-3.2-3b".to_string(),
            model_path: "/models/llama-3.2-3b.Q4_K_M.gguf".to_string(),
            startup_triggered: false,
            server_ready: true,
            model_verified: true,
            llama_server: Some(UnitInfo {
                unit_name: "llama-server".to_string(),
                active_state: "active".to_string(),
                sub_state: "running".to_string(),
                description: "llama.cpp server".to_string(),
            }),
        };

        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["status"], "ok");
        assert_eq!(json["model_alias"], "llama-3.2-3b");
        assert_eq!(json["startup_triggered"], false);
        assert_eq!(json["server_ready"], true);
        assert_eq!(json["model_verified"], true);
        assert_eq!(json["llama_server"]["active_state"], "active");
    }

    #[test]
    fn ensure_started_mismatch_serializes_to_json() {
        let response = EnsureStartedResponse {
            status: HelperStatus::Mismatch,
            message: "expected model path '/models/a.gguf', but server reports '/models/b.gguf'"
                .to_string(),
            model_alias: "model-a".to_string(),
            model_path: "/models/a.gguf".to_string(),
            startup_triggered: false,
            server_ready: true,
            model_verified: false,
            llama_server: Some(UnitInfo {
                unit_name: "llama-server".to_string(),
                active_state: "active".to_string(),
                sub_state: "running".to_string(),
                description: String::new(),
            }),
        };

        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["status"], "mismatch");
        assert_eq!(json["model_verified"], false);
        assert_eq!(json["server_ready"], true);
        assert_eq!(json["model_alias"], "model-a");
    }

    #[test]
    fn ensure_started_error_serializes_to_json() {
        let response = EnsureStartedResponse {
            status: HelperStatus::Error,
            message: "failed to start llama-server: exit code 1".to_string(),
            model_alias: "llama-3.2-3b".to_string(),
            model_path: "/models/llama-3.2-3b.gguf".to_string(),
            startup_triggered: true,
            server_ready: false,
            model_verified: false,
            llama_server: Some(UnitInfo {
                unit_name: "llama-server".to_string(),
                active_state: "failed".to_string(),
                sub_state: "startup_error".to_string(),
                description: String::new(),
            }),
        };

        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["status"], "error");
        assert_eq!(json["startup_triggered"], true);
        assert_eq!(json["llama_server"]["active_state"], "failed");
    }

    #[test]
    fn ensure_started_error_with_null_llama_server() {
        let response = EnsureStartedResponse {
            status: HelperStatus::Error,
            message: "model identity check failed: connection refused".to_string(),
            model_alias: "test-model".to_string(),
            model_path: "/models/test.gguf".to_string(),
            startup_triggered: true,
            server_ready: false,
            model_verified: false,
            llama_server: None,
        };

        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["llama_server"], serde_json::Value::Null);
    }

    #[test]
    fn parses_systemctl_show_output() {
        let output = b"ActiveState=active\nSubState=running\nDescription=llama.cpp server\n";
        let map = parse_systemctl_output(output).unwrap();
        assert_eq!(map.get("ActiveState").unwrap(), "active");
        assert_eq!(map.get("SubState").unwrap(), "running");
        assert_eq!(map.get("Description").unwrap(), "llama.cpp server");
    }

    #[test]
    fn parses_systemctl_output_with_empty_values() {
        let output = b"ActiveState=inactive\nSubState=dead\nDescription=\n";
        let map = parse_systemctl_output(output).unwrap();
        assert_eq!(map.get("ActiveState").unwrap(), "inactive");
        assert_eq!(map.get("Description").unwrap(), "");
    }

    #[test]
    fn helper_config_defaults() {
        let config = HelperConfig::default();
        assert_eq!(config.llama_server_unit, "llama-server");
        assert_eq!(config.inhibit_holder_unit, "llm-wake-proxy-inhibit");
        assert_eq!(config.llama_server_port, 8080);
        assert_eq!(
            config.llama_server_embeddings_unit,
            "llama-server-embeddings"
        );
        assert_eq!(config.llama_server_embeddings_port, 8081);
        assert_eq!(config.server_start_timeout_secs, 60);
    }

    #[test]
    fn helper_config_unit_for_and_port_for() {
        let config = HelperConfig::default();
        assert_eq!(config.unit_for(Target::Chat), "llama-server");
        assert_eq!(
            config.unit_for(Target::Embeddings),
            "llama-server-embeddings"
        );
        assert_eq!(config.port_for(Target::Chat), 8080);
        assert_eq!(config.port_for(Target::Embeddings), 8081);
    }

    #[test]
    fn target_as_str_and_from_str() {
        assert_eq!(Target::Chat.as_str(), "chat");
        assert_eq!(Target::Embeddings.as_str(), "embeddings");
        assert_eq!("chat".parse::<Target>(), Ok(Target::Chat));
        assert_eq!("embeddings".parse::<Target>(), Ok(Target::Embeddings));
        assert!("bogus".parse::<Target>().is_err());
    }

    #[test]
    fn ensure_started_with_already_active_server() {
        let config = HelperConfig {
            server_start_timeout_secs: 1,
            ..HelperConfig::default()
        };

        struct ReadySystemd;

        impl SystemdControl for ReadySystemd {
            fn show_unit(&self, unit: &str) -> Result<HashMap<String, String>, String> {
                let mut map = HashMap::new();
                map.insert("ActiveState".to_string(), "active".to_string());
                map.insert("SubState".to_string(), "running".to_string());
                map.insert("Description".to_string(), format!("{unit} unit"));
                Ok(map)
            }
            fn is_active(&self, _unit: &str) -> Result<bool, String> {
                Ok(true)
            }
            fn start_unit(&self, _unit: &str) -> Result<(), String> {
                Ok(())
            }
        }

        struct ReadyServer;

        impl ServerCheck for ReadyServer {
            fn get_text(&self, _url: &str) -> Result<String, String> {
                Ok(r#"{"object":"list","data":[{"id":"/models/llama-3.2-3b.Q4_K_M.gguf","object":"model","created":1717156800,"owned_by":"llama.cpp"}]}"#.to_string())
            }
        }

        let helper = Helper::with_services(config, Box::new(ReadySystemd), Box::new(ReadyServer));

        let response = helper.ensure_started(
            Target::Chat,
            "llama-3.2-3b",
            "/models/llama-3.2-3b.Q4_K_M.gguf",
        );

        assert_eq!(response.status, HelperStatus::Ok);
        assert!(!response.startup_triggered);
        assert!(response.server_ready);
        assert!(response.model_verified);
        assert_eq!(response.message, "model is already running and ready");
    }

    #[test]
    fn ensure_started_starts_inactive_server_and_verifies_model() {
        let config = HelperConfig {
            server_start_timeout_secs: 1,
            ..HelperConfig::default()
        };

        struct StartOnDemandSystemd {
            active: Arc<Mutex<bool>>,
        }

        impl SystemdControl for StartOnDemandSystemd {
            fn show_unit(&self, unit: &str) -> Result<HashMap<String, String>, String> {
                let mut map = HashMap::new();
                let active = *self.active.lock().unwrap();
                map.insert(
                    "ActiveState".to_string(),
                    if active { "active" } else { "inactive" }.to_string(),
                );
                map.insert(
                    "SubState".to_string(),
                    if active { "running" } else { "dead" }.to_string(),
                );
                map.insert("Description".to_string(), format!("{unit} unit"));
                Ok(map)
            }
            fn is_active(&self, _unit: &str) -> Result<bool, String> {
                let active = *self.active.lock().unwrap();
                Ok(active)
            }
            fn start_unit(&self, _unit: &str) -> Result<(), String> {
                *self.active.lock().unwrap() = true;
                Ok(())
            }
        }

        struct DelayedServer {
            ready: Arc<Mutex<bool>>,
            call_count: Arc<Mutex<u32>>,
        }

        impl ServerCheck for DelayedServer {
            fn get_text(&self, _url: &str) -> Result<String, String> {
                let mut count = self.call_count.lock().unwrap();
                *count += 1;
                if *count > 1 {
                    *self.ready.lock().unwrap() = true;
                }
                if *self.ready.lock().unwrap() {
                    Ok(r#"{"object":"list","data":[{"id":"/models/llama-3.2-3b.Q4_K_M.gguf","object":"model","created":1717156800,"owned_by":"llama.cpp"}]}"#.to_string())
                } else {
                    Err("connection refused".to_string())
                }
            }
        }

        let active = Arc::new(Mutex::new(false));
        let ready = Arc::new(Mutex::new(false));
        let call_count = Arc::new(Mutex::new(0u32));

        let helper = Helper::with_services(
            config,
            Box::new(StartOnDemandSystemd {
                active: active.clone(),
            }),
            Box::new(DelayedServer {
                ready: ready.clone(),
                call_count: call_count.clone(),
            }),
        );

        let response = helper.ensure_started(
            Target::Chat,
            "llama-3.2-3b",
            "/models/llama-3.2-3b.Q4_K_M.gguf",
        );

        assert_eq!(response.status, HelperStatus::Ok);
        assert!(response.startup_triggered);
        assert!(response.server_ready);
        assert!(response.model_verified);
        assert_eq!(response.message, "model started and verified successfully");
    }

    #[test]
    fn ensure_started_detects_model_mismatch() {
        let config = HelperConfig {
            server_start_timeout_secs: 1,
            ..HelperConfig::default()
        };

        struct MismatchSystemd;

        impl SystemdControl for MismatchSystemd {
            fn show_unit(&self, unit: &str) -> Result<HashMap<String, String>, String> {
                let mut map = HashMap::new();
                map.insert("ActiveState".to_string(), "active".to_string());
                map.insert("SubState".to_string(), "running".to_string());
                map.insert("Description".to_string(), format!("{unit} unit"));
                Ok(map)
            }
            fn is_active(&self, _unit: &str) -> Result<bool, String> {
                Ok(true)
            }
            fn start_unit(&self, _unit: &str) -> Result<(), String> {
                Ok(())
            }
        }

        struct WrongModelServer;

        impl ServerCheck for WrongModelServer {
            fn get_text(&self, _url: &str) -> Result<String, String> {
                Ok(r#"{"object":"list","data":[{"id":"/models/llama-3.1-8b.Q4_K_M.gguf","object":"model","created":1717156800,"owned_by":"llama.cpp"}]}"#.to_string())
            }
        }

        let helper = Helper::with_services(
            config,
            Box::new(MismatchSystemd),
            Box::new(WrongModelServer),
        );

        let response = helper.ensure_started(
            Target::Chat,
            "llama-3.2-3b",
            "/models/llama-3.2-3b.Q4_K_M.gguf",
        );

        assert_eq!(response.status, HelperStatus::Mismatch);
        assert!(response.server_ready);
        assert!(!response.model_verified);
        assert!(response.message.contains("mismatch") || response.message.contains("expected"));
    }

    #[test]
    fn ensure_started_reports_startup_failure() {
        let config = HelperConfig::default();

        struct FailingSystemd;

        impl SystemdControl for FailingSystemd {
            fn show_unit(&self, unit: &str) -> Result<HashMap<String, String>, String> {
                let mut map = HashMap::new();
                map.insert("ActiveState".to_string(), "failed".to_string());
                map.insert("SubState".to_string(), "failed".to_string());
                map.insert("Description".to_string(), format!("{unit} unit"));
                Ok(map)
            }
            fn is_active(&self, _unit: &str) -> Result<bool, String> {
                Ok(false)
            }
            fn start_unit(&self, _unit: &str) -> Result<(), String> {
                Err("unit failed to start: exit code 1".to_string())
            }
        }

        struct UnusedServer;

        impl ServerCheck for UnusedServer {
            fn get_text(&self, _url: &str) -> Result<String, String> {
                Err("should not be called".to_string())
            }
        }

        let helper =
            Helper::with_services(config, Box::new(FailingSystemd), Box::new(UnusedServer));

        let response = helper.ensure_started(Target::Chat, "test-model", "/models/test.gguf");

        assert_eq!(response.status, HelperStatus::Error);
        assert!(!response.startup_triggered);
        assert!(!response.server_ready);
        assert!(response.message.contains("failed to start"));
    }

    #[test]
    fn ensure_started_reports_timeout_when_server_never_ready() {
        let config = HelperConfig {
            server_start_timeout_secs: 0,
            ..HelperConfig::default()
        };

        struct StuckSystemd;

        impl SystemdControl for StuckSystemd {
            fn show_unit(&self, unit: &str) -> Result<HashMap<String, String>, String> {
                let mut map = HashMap::new();
                map.insert("ActiveState".to_string(), "activating".to_string());
                map.insert("SubState".to_string(), "start".to_string());
                map.insert("Description".to_string(), format!("{unit} unit"));
                Ok(map)
            }
            fn is_active(&self, _unit: &str) -> Result<bool, String> {
                Ok(false)
            }
            fn start_unit(&self, _unit: &str) -> Result<(), String> {
                Ok(())
            }
        }

        struct NeverReadyServer;

        impl ServerCheck for NeverReadyServer {
            fn get_text(&self, _url: &str) -> Result<String, String> {
                Err("connection refused".to_string())
            }
        }

        let helper =
            Helper::with_services(config, Box::new(StuckSystemd), Box::new(NeverReadyServer));

        let response = helper.ensure_started(Target::Chat, "test-model", "/models/test.gguf");

        assert_eq!(response.status, HelperStatus::Error);
        assert!(!response.server_ready);
        assert!(response.message.contains("did not become ready"));
    }

    // ============= Lease tests =============

    #[test]
    fn lease_acquire_response_serializes_to_json() {
        let response = LeaseAcquireResponse {
            status: HelperStatus::Ok,
            message: "lease renewed".to_string(),
            inhibit_holder: Some(UnitInfo {
                unit_name: "llm-wake-proxy-inhibit".to_string(),
                active_state: "active".to_string(),
                sub_state: "running".to_string(),
                description: "llm-wake-proxy inhibit holder".to_string(),
            }),
            ttl_secs: 900,
            renewed: true,
        };

        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["status"], "ok");
        assert_eq!(json["ttl_secs"], 900);
        assert_eq!(json["renewed"], true);
        assert_eq!(json["inhibit_holder"]["active_state"], "active");
    }

    #[test]
    fn lease_acquire_response_with_error() {
        let response = LeaseAcquireResponse {
            status: HelperStatus::Error,
            message: "failed to acquire lease: systemd-run not found".to_string(),
            inhibit_holder: None,
            ttl_secs: 900,
            renewed: false,
        };

        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["status"], "error");
        assert_eq!(json["inhibit_holder"], serde_json::Value::Null);
    }

    #[test]
    fn lease_release_response_serializes_to_json() {
        let response = LeaseReleaseResponse {
            status: HelperStatus::Ok,
            message: "lease released".to_string(),
            inhibit_holder: Some(UnitInfo {
                unit_name: "llm-wake-proxy-inhibit".to_string(),
                active_state: "inactive".to_string(),
                sub_state: "dead".to_string(),
                description: String::new(),
            }),
        };

        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["status"], "ok");
        assert_eq!(json["inhibit_holder"]["active_state"], "inactive");
    }

    #[test]
    fn lease_release_response_with_error() {
        let response = LeaseReleaseResponse {
            status: HelperStatus::Error,
            message: "failed to release lease: unit not found".to_string(),
            inhibit_holder: None,
        };

        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["status"], "error");
        assert_eq!(json["inhibit_holder"], serde_json::Value::Null);
    }

    #[test]
    fn lease_inspect_response_serializes_to_json() {
        let response = LeaseInspectResponse {
            status: HelperStatus::Ok,
            message: "current lease state".to_string(),
            inhibit_holder: Some(UnitInfo {
                unit_name: "llm-wake-proxy-inhibit".to_string(),
                active_state: "active".to_string(),
                sub_state: "running".to_string(),
                description: "llm-wake-proxy inhibit holder".to_string(),
            }),
        };

        let json = serde_json::to_value(&response).unwrap();
        assert_eq!(json["status"], "ok");
        assert_eq!(json["inhibit_holder"]["active_state"], "active");
    }

    #[test]
    fn lease_acquire_creates_transient_unit() {
        let config = HelperConfig::default();

        struct LeaseSystemd;

        impl SystemdControl for LeaseSystemd {
            fn show_unit(&self, unit: &str) -> Result<HashMap<String, String>, String> {
                let mut map = HashMap::new();
                map.insert("ActiveState".to_string(), "active".to_string());
                map.insert("SubState".to_string(), "running".to_string());
                map.insert("Description".to_string(), format!("{unit} unit"));
                Ok(map)
            }
            fn is_active(&self, _unit: &str) -> Result<bool, String> {
                Ok(false)
            }
            fn start_unit(&self, _unit: &str) -> Result<(), String> {
                Ok(())
            }
            fn replace_transient_unit(
                &self,
                _unit: &str,
                _description: &str,
                _args: &[&str],
            ) -> Result<(), String> {
                Ok(())
            }
        }

        impl ServerCheck for LeaseSystemd {
            fn get_text(&self, _url: &str) -> Result<String, String> {
                Ok(String::new())
            }
        }

        let helper = Helper::with_services(config, Box::new(LeaseSystemd), Box::new(LeaseSystemd));
        let response = helper.lease_acquire(900);

        assert_eq!(response.status, HelperStatus::Ok);
        assert_eq!(response.message, "lease acquired");
        assert_eq!(response.ttl_secs, 900);
        assert!(!response.renewed);
    }

    #[test]
    fn lease_acquire_renews_existing_lease() {
        let config = HelperConfig::default();

        struct RenewLeaseSystemd;

        impl SystemdControl for RenewLeaseSystemd {
            fn show_unit(&self, unit: &str) -> Result<HashMap<String, String>, String> {
                let mut map = HashMap::new();
                map.insert("ActiveState".to_string(), "active".to_string());
                map.insert("SubState".to_string(), "running".to_string());
                map.insert("Description".to_string(), format!("{unit} unit"));
                Ok(map)
            }
            fn is_active(&self, _unit: &str) -> Result<bool, String> {
                Ok(true)
            }
            fn start_unit(&self, _unit: &str) -> Result<(), String> {
                Ok(())
            }
            fn replace_transient_unit(
                &self,
                _unit: &str,
                _description: &str,
                _args: &[&str],
            ) -> Result<(), String> {
                Ok(())
            }
        }

        impl ServerCheck for RenewLeaseSystemd {
            fn get_text(&self, _url: &str) -> Result<String, String> {
                Ok(String::new())
            }
        }

        let helper = Helper::with_services(
            config,
            Box::new(RenewLeaseSystemd),
            Box::new(RenewLeaseSystemd),
        );
        let response = helper.lease_acquire(600);

        assert_eq!(response.status, HelperStatus::Ok);
        assert_eq!(response.message, "lease renewed");
        assert!(response.renewed);
    }

    #[test]
    fn lease_acquire_reports_failure() {
        let config = HelperConfig::default();

        struct FailingLeaseSystemd;

        impl SystemdControl for FailingLeaseSystemd {
            fn show_unit(&self, unit: &str) -> Result<HashMap<String, String>, String> {
                let mut map = HashMap::new();
                map.insert("ActiveState".to_string(), "inactive".to_string());
                map.insert("SubState".to_string(), "dead".to_string());
                map.insert("Description".to_string(), format!("{unit} unit"));
                Ok(map)
            }
            fn is_active(&self, _unit: &str) -> Result<bool, String> {
                Ok(false)
            }
            fn start_unit(&self, _unit: &str) -> Result<(), String> {
                Ok(())
            }
            fn replace_transient_unit(
                &self,
                _unit: &str,
                _description: &str,
                _args: &[&str],
            ) -> Result<(), String> {
                Err("systemd-run binary not found".to_string())
            }
        }

        impl ServerCheck for FailingLeaseSystemd {
            fn get_text(&self, _url: &str) -> Result<String, String> {
                Ok(String::new())
            }
        }

        let helper = Helper::with_services(
            config,
            Box::new(FailingLeaseSystemd),
            Box::new(FailingLeaseSystemd),
        );
        let response = helper.lease_acquire(900);

        assert_eq!(response.status, HelperStatus::Error);
        assert!(response.message.contains("systemd-run"));
        assert!(!response.renewed);
    }

    #[test]
    fn lease_release_stops_inhibit_unit() {
        let config = HelperConfig::default();

        struct StoppableSystemd;

        impl SystemdControl for StoppableSystemd {
            fn show_unit(&self, unit: &str) -> Result<HashMap<String, String>, String> {
                let mut map = HashMap::new();
                map.insert("ActiveState".to_string(), "inactive".to_string());
                map.insert("SubState".to_string(), "dead".to_string());
                map.insert("Description".to_string(), format!("{unit} unit"));
                Ok(map)
            }
            fn is_active(&self, _unit: &str) -> Result<bool, String> {
                Ok(false)
            }
            fn start_unit(&self, _unit: &str) -> Result<(), String> {
                Ok(())
            }
            fn stop_unit(&self, _unit: &str) -> Result<(), String> {
                Ok(())
            }
        }

        impl ServerCheck for StoppableSystemd {
            fn get_text(&self, _url: &str) -> Result<String, String> {
                Ok(String::new())
            }
        }

        let helper = Helper::with_services(
            config,
            Box::new(StoppableSystemd),
            Box::new(StoppableSystemd),
        );
        let response = helper.lease_release();

        assert_eq!(response.status, HelperStatus::Ok);
        assert_eq!(response.message, "lease released");
    }

    #[test]
    fn lease_release_reports_failure() {
        let config = HelperConfig::default();

        struct UnstoppableSystemd;

        impl SystemdControl for UnstoppableSystemd {
            fn show_unit(&self, _unit: &str) -> Result<HashMap<String, String>, String> {
                Err("unit not found".to_string())
            }
            fn is_active(&self, _unit: &str) -> Result<bool, String> {
                Ok(false)
            }
            fn start_unit(&self, _unit: &str) -> Result<(), String> {
                Ok(())
            }
            fn stop_unit(&self, _unit: &str) -> Result<(), String> {
                Err("unit not found".to_string())
            }
        }

        impl ServerCheck for UnstoppableSystemd {
            fn get_text(&self, _url: &str) -> Result<String, String> {
                Ok(String::new())
            }
        }

        let helper = Helper::with_services(
            config,
            Box::new(UnstoppableSystemd),
            Box::new(UnstoppableSystemd),
        );
        let response = helper.lease_release();

        assert_eq!(response.status, HelperStatus::Error);
        assert!(response.message.contains("unit not found"));
    }

    #[test]
    fn lease_inspect_returns_current_state() {
        let config = HelperConfig::default();

        struct InspectableSystemd;

        impl SystemdControl for InspectableSystemd {
            fn show_unit(&self, unit: &str) -> Result<HashMap<String, String>, String> {
                let mut map = HashMap::new();
                map.insert("ActiveState".to_string(), "active".to_string());
                map.insert("SubState".to_string(), "running".to_string());
                map.insert("Description".to_string(), format!("{unit} unit"));
                Ok(map)
            }
            fn is_active(&self, _unit: &str) -> Result<bool, String> {
                Ok(true)
            }
            fn start_unit(&self, _unit: &str) -> Result<(), String> {
                Ok(())
            }
        }

        impl ServerCheck for InspectableSystemd {
            fn get_text(&self, _url: &str) -> Result<String, String> {
                Ok(String::new())
            }
        }

        let helper = Helper::with_services(
            config,
            Box::new(InspectableSystemd),
            Box::new(InspectableSystemd),
        );
        let response = helper.lease_inspect();

        assert_eq!(response.status, HelperStatus::Ok);
        let unit = response.inhibit_holder.unwrap();
        assert_eq!(unit.active_state, "active");
    }
}

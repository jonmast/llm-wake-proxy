use std::{env, time::Duration};

#[derive(Clone, Debug)]
pub struct AppConfig {
    pub listen_port: u16,
    pub model: ModelConfig,
    pub embeddings: EmbeddingsConfig,
    pub warm_execution: WarmExecutionConfig,
    pub host: HostConfig,
    pub cold_start_max_waiting: usize,
}

#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub alias: String,
    pub provider_id: String,
    pub owned_by: String,
}

#[derive(Clone, Debug)]
pub struct EmbeddingsConfig {
    pub enabled: bool,
    pub backend: Option<EmbeddingsBackendConfig>,
}

#[derive(Clone, Debug)]
pub struct EmbeddingsBackendConfig {
    pub alias: String,
    pub provider_id: String,
    pub owned_by: String,
    pub model_path: String,
    pub tunnel_local_port: u16,
    pub remote_port: u16,
}

#[derive(Clone, Debug)]
pub struct WarmExecutionConfig {
    pub max_active_requests: usize,
    pub max_queued_requests: usize,
    pub queue_timeout: Duration,
}

#[derive(Clone, Debug)]
pub struct HostConfig {
    pub host: String,
    pub ssh_user: String,
    pub ssh_port: u16,
    pub wol_mac: [u8; 6],
    pub wol_broadcast: String,
    pub wol_port: u16,
    pub helper_path: String,
    pub model_path: String,
    pub ssh_key_path: String,
    pub tunnel_local_port: u16,
    pub remote_port: u16,
}

impl Default for WarmExecutionConfig {
    fn default() -> Self {
        Self {
            max_active_requests: 2,
            max_queued_requests: 16,
            queue_timeout: Duration::from_secs(30),
        }
    }
}

impl AppConfig {
    pub fn from_env() -> Self {
        let model = ModelConfig {
            alias: read_var("MODEL_ALIAS").unwrap_or_else(|| "llm-wake-proxy".to_string()),
            provider_id: read_var("MODEL_PROVIDER_ID").unwrap_or_else(|| "llama.cpp".to_string()),
            owned_by: read_var("MODEL_OWNED_BY").unwrap_or_else(|| "llm-wake-proxy".to_string()),
        };

        let embeddings_backend =
            read_var("EMBEDDINGS_MODEL_PATH").map(|model_path| EmbeddingsBackendConfig {
                alias: read_var("EMBEDDINGS_MODEL_ALIAS")
                    .unwrap_or_else(|| format!("{}-embeddings", model.alias)),
                provider_id: read_var("EMBEDDINGS_MODEL_PROVIDER_ID")
                    .unwrap_or_else(|| model.provider_id.clone()),
                owned_by: read_var("EMBEDDINGS_MODEL_OWNED_BY")
                    .unwrap_or_else(|| model.owned_by.clone()),
                model_path,
                tunnel_local_port: read_var("EMBEDDINGS_TUNNEL_LOCAL_PORT")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(18081),
                remote_port: read_var("EMBEDDINGS_LLAMA_SERVER_PORT")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(8081),
            });

        Self {
            listen_port: read_var("PORT")
                .and_then(|value| value.parse().ok())
                .unwrap_or(3000),
            model,
            embeddings: EmbeddingsConfig {
                // A dedicated embeddings backend implies embeddings are enabled,
                // regardless of EMBEDDINGS_ENABLED (otherwise /v1/models would
                // advertise a model that /v1/embeddings then rejects).
                enabled: embeddings_backend.is_some()
                    || read_var("EMBEDDINGS_ENABLED")
                        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes"))
                        .unwrap_or(true),
                backend: embeddings_backend,
            },
            warm_execution: WarmExecutionConfig {
                max_active_requests: read_positive_usize("WARM_MAX_ACTIVE_REQUESTS", 2),
                max_queued_requests: read_nonnegative_usize("WARM_MAX_QUEUED_REQUESTS", 16),
                queue_timeout: Duration::from_secs(
                    read_var("WARM_QUEUE_TIMEOUT_SECS")
                        .and_then(|value| value.parse().ok())
                        .unwrap_or(30),
                ),
            },
            host: HostConfig {
                host: read_var("SSH_HOST").expect("SSH_HOST must be set"),
                ssh_user: read_var("SSH_USER").expect("SSH_USER must be set"),
                ssh_port: read_var("SSH_PORT")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(22),
                wol_mac: {
                    let mac_str = read_var("WOL_MAC_ADDRESS").expect("WOL_MAC_ADDRESS must be set");
                    parse_mac(&mac_str)
                        .expect("WOL_MAC_ADDRESS must be in XX:XX:XX:XX:XX:XX format")
                },
                wol_broadcast: read_var("WOL_BROADCAST_ADDR")
                    .unwrap_or_else(|| "255.255.255.255".to_string()),
                wol_port: read_var("WOL_PORT")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(9),
                helper_path: read_var("HELPER_PATH")
                    .unwrap_or_else(|| "/usr/local/bin/llm-wake-proxy-helper".to_string()),
                model_path: read_var("MODEL_PATH").expect("MODEL_PATH must be set"),
                ssh_key_path: read_var("SSH_KEY_PATH")
                    .unwrap_or_else(|| "~/.ssh/ssh-privatekey".to_string()),
                tunnel_local_port: read_var("TUNNEL_LOCAL_PORT")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(18080),
                remote_port: read_var("LLAMA_SERVER_PORT")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(8080),
            },
            cold_start_max_waiting: read_positive_usize("COLD_START_MAX_WAITING", 32),
        }
    }
}

fn read_var(name: &str) -> Option<String> {
    env::var(name).ok().filter(|value| !value.trim().is_empty())
}

fn read_positive_usize(name: &str, default: usize) -> usize {
    read_var(name)
        .and_then(|value| value.parse().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn read_nonnegative_usize(name: &str, default: usize) -> usize {
    read_var(name)
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

pub(crate) fn parse_mac(input: &str) -> Option<[u8; 6]> {
    let parts: Vec<&str> = input.split(':').collect();
    if parts.len() != 6 {
        return None;
    }
    let mut mac = [0u8; 6];
    for (i, part) in parts.iter().enumerate() {
        mac[i] = u8::from_str_radix(part, 16).ok()?;
    }
    Some(mac)
}

#[cfg(test)]
impl Default for HostConfig {
    fn default() -> Self {
        Self {
            host: "192.168.1.100".to_string(),
            ssh_user: "test-user".to_string(),
            ssh_port: 22,
            wol_mac: [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF],
            wol_broadcast: "255.255.255.255".to_string(),
            wol_port: 9,
            helper_path: "/usr/local/bin/llm-wake-proxy-helper".to_string(),
            model_path: "/models/test-model.gguf".to_string(),
            ssh_key_path: "~/.ssh/ssh-privatekey".to_string(),
            tunnel_local_port: 18080,
            remote_port: 8080,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Mutex, OnceLock};

    use super::*;

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn set_required_host_env() {
        unsafe {
            env::set_var("SSH_HOST", "test-host");
            env::set_var("SSH_USER", "test-user");
            env::set_var("WOL_MAC_ADDRESS", "AA:BB:CC:DD:EE:FF");
            env::set_var("MODEL_PATH", "/models/test.gguf");
        }
    }

    fn remove_required_host_env() {
        unsafe {
            env::remove_var("SSH_HOST");
            env::remove_var("SSH_USER");
            env::remove_var("WOL_MAC_ADDRESS");
            env::remove_var("MODEL_PATH");
        }
    }

    #[test]
    fn default_warm_execution_limits_are_positive() {
        let _guard = env_lock().lock().unwrap();
        set_required_host_env();
        unsafe {
            env::remove_var("WARM_MAX_ACTIVE_REQUESTS");
            env::remove_var("WARM_MAX_QUEUED_REQUESTS");
        }

        let config = AppConfig::from_env();
        assert!(config.warm_execution.max_active_requests > 0);
        assert!(config.warm_execution.max_queued_requests > 0);
        remove_required_host_env();
    }

    #[test]
    fn queue_limit_allows_zero_while_active_limit_stays_positive() {
        let _guard = env_lock().lock().unwrap();
        set_required_host_env();
        unsafe {
            env::set_var("WARM_MAX_ACTIVE_REQUESTS", "0");
            env::set_var("WARM_MAX_QUEUED_REQUESTS", "0");
        }

        let config = AppConfig::from_env();

        assert_eq!(config.warm_execution.max_active_requests, 2);
        assert_eq!(config.warm_execution.max_queued_requests, 0);

        unsafe {
            env::remove_var("WARM_MAX_ACTIVE_REQUESTS");
            env::remove_var("WARM_MAX_QUEUED_REQUESTS");
        }
        remove_required_host_env();
    }

    #[test]
    fn parse_mac_accepts_colon_format() {
        let mac = parse_mac("AA:BB:CC:DD:EE:FF").unwrap();
        assert_eq!(mac, [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF]);
    }

    #[test]
    fn parse_mac_rejects_short_input() {
        assert!(parse_mac("AA:BB:CC:DD:EE").is_none());
    }

    #[test]
    fn parse_mac_rejects_invalid_hex() {
        assert!(parse_mac("AA:BB:CC:DD:EE:GG").is_none());
    }

    #[test]
    fn embeddings_backend_is_none_when_model_path_unset() {
        let _guard = env_lock().lock().unwrap();
        set_required_host_env();
        unsafe {
            env::remove_var("EMBEDDINGS_MODEL_PATH");
        }

        let config = AppConfig::from_env();
        assert!(config.embeddings.backend.is_none());

        remove_required_host_env();
    }

    #[test]
    fn embeddings_backend_uses_defaults_when_only_model_path_set() {
        let _guard = env_lock().lock().unwrap();
        set_required_host_env();
        unsafe {
            env::set_var("EMBEDDINGS_MODEL_PATH", "/models/embed.gguf");
            env::remove_var("EMBEDDINGS_MODEL_ALIAS");
            env::remove_var("EMBEDDINGS_MODEL_PROVIDER_ID");
            env::remove_var("EMBEDDINGS_MODEL_OWNED_BY");
            env::remove_var("EMBEDDINGS_TUNNEL_LOCAL_PORT");
            env::remove_var("EMBEDDINGS_LLAMA_SERVER_PORT");
        }

        let config = AppConfig::from_env();
        let backend = config.embeddings.backend.expect("backend should be set");

        assert_eq!(backend.model_path, "/models/embed.gguf");
        assert_eq!(backend.alias, format!("{}-embeddings", config.model.alias));
        assert_eq!(backend.provider_id, config.model.provider_id);
        assert_eq!(backend.owned_by, config.model.owned_by);
        assert_eq!(backend.tunnel_local_port, 18081);
        assert_eq!(backend.remote_port, 8081);

        unsafe {
            env::remove_var("EMBEDDINGS_MODEL_PATH");
        }
        remove_required_host_env();
    }

    #[test]
    fn embeddings_backend_honors_overrides() {
        let _guard = env_lock().lock().unwrap();
        set_required_host_env();
        unsafe {
            env::set_var("EMBEDDINGS_MODEL_PATH", "/models/embed.gguf");
            env::set_var("EMBEDDINGS_MODEL_ALIAS", "embed-alias");
            env::set_var("EMBEDDINGS_MODEL_PROVIDER_ID", "custom-provider");
            env::set_var("EMBEDDINGS_MODEL_OWNED_BY", "custom-owner");
            env::set_var("EMBEDDINGS_TUNNEL_LOCAL_PORT", "28081");
            env::set_var("EMBEDDINGS_LLAMA_SERVER_PORT", "9090");
        }

        let config = AppConfig::from_env();
        let backend = config.embeddings.backend.expect("backend should be set");

        assert_eq!(backend.alias, "embed-alias");
        assert_eq!(backend.provider_id, "custom-provider");
        assert_eq!(backend.owned_by, "custom-owner");
        assert_eq!(backend.tunnel_local_port, 28081);
        assert_eq!(backend.remote_port, 9090);

        unsafe {
            env::remove_var("EMBEDDINGS_MODEL_PATH");
            env::remove_var("EMBEDDINGS_MODEL_ALIAS");
            env::remove_var("EMBEDDINGS_MODEL_PROVIDER_ID");
            env::remove_var("EMBEDDINGS_MODEL_OWNED_BY");
            env::remove_var("EMBEDDINGS_TUNNEL_LOCAL_PORT");
            env::remove_var("EMBEDDINGS_LLAMA_SERVER_PORT");
        }
        remove_required_host_env();
    }

    #[test]
    fn embeddings_backend_forces_enabled_even_if_explicitly_disabled() {
        let _guard = env_lock().lock().unwrap();
        set_required_host_env();
        unsafe {
            env::set_var("EMBEDDINGS_MODEL_PATH", "/models/embed.gguf");
            env::set_var("EMBEDDINGS_ENABLED", "false");
        }

        let config = AppConfig::from_env();
        assert!(config.embeddings.enabled);
        assert!(config.embeddings.backend.is_some());

        unsafe {
            env::remove_var("EMBEDDINGS_MODEL_PATH");
            env::remove_var("EMBEDDINGS_ENABLED");
        }
        remove_required_host_env();
    }
}

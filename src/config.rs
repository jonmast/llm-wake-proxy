use std::{env, time::Duration};

#[derive(Clone, Debug)]
pub struct AppConfig {
    pub listen_port: u16,
    pub model: ModelConfig,
    pub embeddings: EmbeddingsConfig,
    pub warm_execution: WarmExecutionConfig,
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
}

#[derive(Clone, Debug)]
pub struct WarmExecutionConfig {
    pub max_active_requests: usize,
    pub max_queued_requests: usize,
    pub queue_timeout: Duration,
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
        Self {
            listen_port: read_var("PORT")
                .and_then(|value| value.parse().ok())
                .unwrap_or(3000),
            model: ModelConfig {
                alias: read_var("MODEL_ALIAS").unwrap_or_else(|| "llm-wake-proxy".to_string()),
                provider_id: read_var("MODEL_PROVIDER_ID")
                    .unwrap_or_else(|| "llama.cpp".to_string()),
                owned_by: read_var("MODEL_OWNED_BY")
                    .unwrap_or_else(|| "llm-wake-proxy".to_string()),
            },
            embeddings: EmbeddingsConfig {
                enabled: read_var("EMBEDDINGS_ENABLED")
                    .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes"))
                    .unwrap_or(true),
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

#[cfg(test)]
mod tests {
    use std::sync::{Mutex, OnceLock};

    use super::*;

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn default_warm_execution_limits_are_positive() {
        let _guard = env_lock().lock().unwrap();
        unsafe {
            env::remove_var("WARM_MAX_ACTIVE_REQUESTS");
            env::remove_var("WARM_MAX_QUEUED_REQUESTS");
        }

        let config = AppConfig::from_env();
        assert!(config.warm_execution.max_active_requests > 0);
        assert!(config.warm_execution.max_queued_requests > 0);
    }

    #[test]
    fn queue_limit_allows_zero_while_active_limit_stays_positive() {
        let _guard = env_lock().lock().unwrap();
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
    }
}

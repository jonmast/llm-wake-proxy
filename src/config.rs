use std::env;

#[derive(Clone, Debug)]
pub struct AppConfig {
    pub listen_port: u16,
    pub model: ModelConfig,
    pub embeddings: EmbeddingsConfig,
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
        }
    }
}

fn read_var(name: &str) -> Option<String> {
    env::var(name).ok().filter(|value| !value.trim().is_empty())
}

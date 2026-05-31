use std::net::SocketAddr;

use llm_wake_proxy::{app::build_router, config::AppConfig, state::AppState};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_target(false)
        .compact()
        .init();

    let config = AppConfig::from_env();
    let app = build_router(AppState::new(config.clone()));
    let addr = SocketAddr::from(([0, 0, 0, 0], config.listen_port));
    let listener = tokio::net::TcpListener::bind(addr).await?;

    tracing::info!(address = %addr, model_alias = %config.model.alias, "listening");
    axum::serve(listener, app).await?;

    Ok(())
}

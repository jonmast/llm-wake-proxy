use std::process::ExitCode;

use llm_wake_proxy::helper::{Helper, HelperConfig, HelperStatus};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();

    let subcommand = args.get(1).map(|s| s.as_str());

    let config = HelperConfig::from_env();
    let helper = Helper::new(config);

    match subcommand {
        Some("status") => {
            let response = helper.status();
            println!("{}", serde_json::to_string(&response).unwrap());
            if matches!(response.status, HelperStatus::Ok) {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Some("ensure-started") => {
            let model_alias = match args.get(2) {
                Some(alias) => alias,
                None => {
                    eprintln!("Usage: helper ensure-started <model-alias>");
                    eprintln!("");
                    eprintln!("Environment:");
                    eprintln!("  EXPECTED_MODEL_PATH    Path to verify against running server");
                    eprintln!("  LLAMA_SERVER_UNIT      systemd unit name (default: llama-server)");
                    eprintln!(
                        "  LLAMA_SERVER_PORT      Port for server health check (default: 8080)"
                    );
                    eprintln!(
                        "  SERVER_START_TIMEOUT_SECS  How long to wait for server (default: 60)"
                    );
                    return ExitCode::FAILURE;
                }
            };

            let model_path = match std::env::var("EXPECTED_MODEL_PATH") {
                Ok(path) if !path.is_empty() => path,
                _ => {
                    eprintln!("EXPECTED_MODEL_PATH must be set for ensure-started");
                    return ExitCode::FAILURE;
                }
            };

            let response = helper.ensure_started(model_alias, &model_path);
            println!("{}", serde_json::to_string(&response).unwrap());

            match response.status {
                HelperStatus::Ok => ExitCode::SUCCESS,
                HelperStatus::Mismatch => {
                    eprintln!("Model mismatch: {}", response.message);
                    ExitCode::FAILURE
                }
                HelperStatus::Error => {
                    eprintln!("Error: {}", response.message);
                    ExitCode::FAILURE
                }
            }
        }
        _ => {
            eprintln!("Usage:");
            eprintln!("  helper status                    Show host unit and model status");
            eprintln!("  helper ensure-started <alias>    Start and verify model server");
            eprintln!("");
            eprintln!("Environment:");
            eprintln!("  EXPECTED_MODEL_PATH    Path to the model file for verification");
            eprintln!("  LLAMA_SERVER_UNIT      systemd unit name (default: llama-server)");
            eprintln!(
                "  INHIBIT_HOLDER_UNIT    systemd unit name (default: llm-wake-proxy-inhibit)"
            );
            eprintln!("  LLAMA_SERVER_PORT      Port for server health check (default: 8080)");
            eprintln!("  SERVER_START_TIMEOUT_SECS  How long to wait for server (default: 60)");
            ExitCode::FAILURE
        }
    }
}

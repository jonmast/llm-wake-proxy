use std::process::ExitCode;

use llm_wake_proxy::helper::{Helper, HelperConfig, HelperStatus};

fn print_json<T: serde::Serialize>(response: &T) {
    println!("{}", serde_json::to_string(response).unwrap());
}

fn exit_for_status(status: &HelperStatus, stderr_msg: Option<&str>) -> ExitCode {
    match status {
        HelperStatus::Ok => ExitCode::SUCCESS,
        _ => {
            if let Some(msg) = stderr_msg {
                eprintln!("{msg}");
            }
            ExitCode::FAILURE
        }
    }
}

fn parse_ttl_arg(args: &[String], pos: usize) -> u64 {
    if let Some(flag) = args.get(pos) {
        if flag == "--ttl" {
            if let Some(value) = args.get(pos + 1) {
                if let Ok(ttl) = value.parse::<u64>() {
                    return ttl;
                }
                eprintln!(
                    "Warning: --ttl value '{}' is not a valid number, using default 900",
                    value
                );
            } else {
                eprintln!("Warning: --ttl requires a value, using default 900");
            }
        } else {
            eprintln!("Warning: unexpected argument '{}', use --ttl <secs>", flag);
        }
    }
    900
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();

    let subcommand = args.get(1).map(|s| s.as_str());

    let config = HelperConfig::from_env();
    let helper = Helper::new(config);

    match subcommand {
        Some("status") => {
            let response = helper.status();
            print_json(&response);
            exit_for_status(&response.status, None)
        }
        Some("ensure-started") => {
            let model_alias = match args.get(2) {
                Some(alias) => alias,
                None => {
                    eprintln!("Usage: helper ensure-started <model-alias>");
                    eprintln!();
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
            print_json(&response);

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
        Some("lease") => {
            let lease_cmd = args.get(2).map(|s| s.as_str());
            match lease_cmd {
                Some("acquire") => {
                    let ttl = parse_ttl_arg(&args, 3);

                    let response = helper.lease_acquire(ttl);
                    print_json(&response);
                    exit_for_status(&response.status, Some(&response.message))
                }
                Some("release") => {
                    let response = helper.lease_release();
                    print_json(&response);
                    exit_for_status(&response.status, Some(&response.message))
                }
                Some("inspect") => {
                    let response = helper.lease_inspect();
                    print_json(&response);
                    exit_for_status(&response.status, None)
                }
                _ => {
                    eprintln!("Usage: helper lease <acquire|release|inspect>");
                    eprintln!();
                    eprintln!("  acquire [--ttl <secs>]    Create or renew inhibit lease");
                    eprintln!("  release                  Stop inhibit lease");
                    eprintln!("  inspect                  Show current lease state");
                    eprintln!();
                    eprintln!("Environment:");
                    eprintln!("  INHIBIT_HOLDER_UNIT    systemd unit name (default: llm-wake-proxy-inhibit)");
                    ExitCode::FAILURE
                }
            }
        }
        _ => {
            eprintln!("Usage:");
            eprintln!("  helper status                    Show host unit and model status");
            eprintln!("  helper ensure-started <alias>    Start and verify model server");
            eprintln!("  helper lease <acquire|release|inspect>  Manage inhibit lease");
            eprintln!();
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

//! Script-based lifecycle hooks for model management.
//!
//! All model lifecycle operations (wake, sleep, liveness) are delegated to
//! user-provided scripts. llmux does not know or care how models are started,
//! stopped, or health-checked — that's entirely up to the scripts.
//!
//! Hooks are executed via `sh -c`, so they can be either a path to an
//! executable or an inline shell script.

use crate::config::ModelConfig;
use std::collections::HashMap;
use tokio::process::Command;
use tracing::{debug, info, warn};

/// The name of the model.
const LLMUX_ENV_MODEL: &str = "LLMUX_MODEL";

/// The destination port for the model entry.
const LLMUX_ENV_DEST_PORT: &str = "LLMUX_DEST_PORT";

/// Errors from hook script execution
#[derive(Debug, thiserror::Error)]
pub enum HookError {
    #[error("hook {hook} failed for model {model} (exit code {code}): {stderr}")]
    Failed {
        model: String,
        hook: String,
        code: i32,
        stderr: String,
    },

    #[error("hook execution error: {0}")]
    Io(#[from] std::io::Error),

    #[error("model not found: {0}")]
    ModelNotFound(String),
}

/// Runs lifecycle scripts for models.
pub struct HookRunner {
    configs: HashMap<String, ModelConfig>,
}

impl HookRunner {
    pub fn new(configs: HashMap<String, ModelConfig>) -> Self {
        Self { configs }
    }

    pub fn registered_models(&self) -> Vec<String> {
        self.configs.keys().cloned().collect()
    }

    pub fn model_port(&self, model: &str) -> Option<u16> {
        self.configs.get(model).map(|c| c.port)
    }

    pub fn is_registered(&self, model: &str) -> bool {
        self.configs.contains_key(model)
    }

    /// Run the wake script for a model. Returns Ok(()) when the model is ready.
    ///
    /// The wake script must be idempotent — it should bring a model from any
    /// state (stopped, sleeping, already running) to a running state.
    pub async fn run_wake(&self, model: &str) -> Result<(), HookError> {
        let config = self
            .configs
            .get(model)
            .ok_or_else(|| HookError::ModelNotFound(model.to_string()))?;
        run_hook(&config.wake, model, config.port, "wake").await
    }

    /// Run the sleep script for a model. Returns Ok(()) when the model is asleep.
    pub async fn run_sleep(&self, model: &str) -> Result<(), HookError> {
        let config = self
            .configs
            .get(model)
            .ok_or_else(|| HookError::ModelNotFound(model.to_string()))?;
        run_hook(&config.sleep, model, config.port, "sleep").await
    }

    /// Run the alive script for a model. Returns true if healthy, false if not.
    ///
    /// Only returns Err for execution failures (script not found, permission
    /// denied, etc.), not for a non-zero exit code (which means "unhealthy").
    pub async fn run_alive(&self, model: &str) -> Result<bool, HookError> {
        let config = self
            .configs
            .get(model)
            .ok_or_else(|| HookError::ModelNotFound(model.to_string()))?;
        match run_hook(&config.alive, model, config.port, "alive").await {
            Ok(()) => Ok(true),
            Err(HookError::Failed { .. }) => Ok(false),
            Err(e) => Err(e),
        }
    }
}

async fn run_hook(script: &str, model: &str, port: u16, hook_name: &str) -> Result<(), HookError> {
    debug!(model = %model, hook = %hook_name, "Running hook");

    let start = std::time::Instant::now();

    let port_s = port.to_string();
    let output = Command::new("sh")
        .arg("-c")
        .arg(script)
        .env(LLMUX_ENV_MODEL, model)
        .env(LLMUX_ENV_DEST_PORT, &port_s)
        .output()
        .await
        .map_err(HookError::Io)?;

    let duration = start.elapsed();
    metrics::histogram!(
        "llmux_hook_duration_seconds",
        "model" => model.to_string(),
        "hook" => hook_name.to_string()
    )
    .record(duration.as_secs_f64());

    if !output.stdout.is_empty() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        debug!(model = %model, hook = %hook_name, stdout = %stdout.trim_end(), "Hook stdout");
    }

    if !output.stderr.is_empty() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        if output.status.success() {
            debug!(model = %model, hook = %hook_name, stderr = %stderr.trim_end(), "Hook stderr");
        } else {
            warn!(model = %model, hook = %hook_name, stderr = %stderr.trim_end(), "Hook failed");
        }
    }

    if !output.status.success() {
        metrics::counter!(
            "llmux_hook_failures_total",
            "model" => model.to_string(),
            "hook" => hook_name.to_string()
        )
        .increment(1);

        let code = output.status.code().unwrap_or(-1);
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        info!(
            event = "hook_failed",
            model = %model,
            hook = %hook_name,
            exit_code = code,
            duration_secs = duration.as_secs_f64(),
            stderr = %stderr.trim_end(),
            "Hook failed"
        );
        return Err(HookError::Failed {
            model: model.to_string(),
            hook: hook_name.to_string(),
            code,
            stderr,
        });
    }

    info!(
        event = "hook_completed",
        model = %model,
        hook = %hook_name,
        duration_secs = duration.as_secs_f64(),
        "Hook completed"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn hook_scripts_receive_environment() {
        // 1. the hook script here uses only builtin commands from sh (no additional dependencies introduced).
        // 2. validation relies on returning the environment through stderr
        let res = run_hook("printenv >&2; exit 300", "dummy-1B", 1999u16, "testhook")
            .await
            .expect_err("should fail");

        let env_str = match res {
            HookError::Failed {
                model,
                hook,
                code,
                stderr,
            } => {
                assert_eq!(model, "dummy-1B");
                assert_eq!(hook, "testhook");
                assert_eq!(code, 300 % 256);
                stderr
            }
            x => {
                panic!("expected HookError::Failed but got {x:?}");
            }
        };

        let lines: Vec<String> = env_str.split("\n").map(String::from).collect();

        // the child environment receives this cargo environment variable if and only if
        // it's passed down from the parent.
        lines
            .iter()
            .find(|v| v.starts_with("CARGO_PKG_VERSION="))
            .expect("parent environment should be passed down.");

        // using hardcoded constants for stability.
        lines
            .iter()
            .find(|v| *v == "LLMUX_MODEL=dummy-1B")
            .expect("llmux must set LLMUX_MODEL to the model name");

        lines
            .iter()
            .find(|v| *v == "LLMUX_DEST_PORT=1999")
            .expect("llmux must set LLMUX_DEST_PORT to the port onto which the model is served");
    }

    #[tokio::test]
    async fn hook_script_tolerates_garbage_output() {
        // \200 is octal sequence for 0x80, which should be part of a 2-byte codepoint
        run_hook("echo 'not utf\\200-8$'; exit 0", "model", 10u16, "testhook")
            .await
            .expect("non-utf8 output should be parsed gracefully");

        run_hook(
            "echo 'not utf\\200-8$' >&2; exit 0",
            "model",
            10u16,
            "testhook",
        )
        .await
        .expect("non-utf8 output should be parsed gracefully");

        // the stderr returned on a failure is lossy when error is not utf-8
        let out = run_hook(
            "echo 'not utf\\200-8' >&2; exit 3",
            "model",
            10u16,
            "testhook",
        )
        .await
        .expect_err("exit 1 should trigger error");

        match out {
            HookError::Failed {
                model: _,
                hook: _,
                code,
                stderr,
            } => {
                assert_eq!(code, 3);
                // bad codes replaced with U+FFFD
                assert_eq!("not utf\u{fffd}-8\n", stderr);
            }
            err => panic!("unexpected error: {err:?}"),
        };
    }

    #[tokio::test]
    async fn hook_script_omits_stdout_on_error() {
        // \200 is octal sequence for 0x80, which should introduce a 2-byte codepoint.
        let out = run_hook("echo out; echo err >&2; exit 1", "model", 10u16, "testhook")
            .await
            .expect_err("exit 1 should trigger a failure");

        match out {
            HookError::Failed {
                model: _,
                hook: _,
                code,
                stderr,
            } => {
                assert_eq!(code, 1);
                // stdout is not part of the mix
                assert_eq!("err\n", stderr);
            }
            err => panic!("unexpected error: {err:?}"),
        };
    }
}

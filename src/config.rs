//! Configuration for llmux v2.
//!
//! All model lifecycle management is delegated to user-provided scripts.
//! llmux only handles request routing, draining, and policy decisions.
//!
//! Supports both YAML and JSON config files (detected by extension).

use anyhow::{Context, Result, anyhow};
use http::Uri;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::IpAddr;
use std::path::Path;
use std::time::Duration;

/// Top-level configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Models to manage
    pub models: HashMap<String, ModelConfig>,

    /// Switch policy configuration
    #[serde(default)]
    pub policy: PolicyConfig,

    /// Proxy port
    #[serde(default = "default_port")]
    pub port: u16,
}

/// Configuration for a single model.
///
/// Each model is defined by a target port (where the inference server listens)
/// and three lifecycle hooks. Hooks can be either a path to an executable script
/// or an inline shell script:
///
/// ```yaml
/// models:
///   llama:
///     port: 8001
///     wake: ./scripts/wake-llama.sh
///     sleep: |
///       kill $(cat /tmp/llama.pid)
///       rm /tmp/llama.pid
///     alive: curl -sf http://localhost:8001/health
/// ```
///
/// All hooks are executed via `sh -c` with LLMUX_MODEL set in the environment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Port on which the model's inference server listens.
    pub port: u16,

    /// Hostname/address on which the model's inference server listens.
    #[serde(default = "default_host")]
    pub host: String,

    /// Script to bring the model to a running state (idempotent).
    /// Can be a path to an executable or an inline shell script.
    /// Called with LLMUX_* env vars set to the model configuration.
    /// Must exit 0 when the model is ready to serve requests.
    pub wake: String,

    /// Script to put the model to sleep / free resources.
    /// Can be a path to an executable or an inline shell script.
    /// Called with LLMUX_* env vars set to the model configuration.
    /// Must exit 0 when the model is fully stopped/sleeping.
    pub sleep: String,

    /// Script to check if the model is alive and healthy.
    /// Can be a path to an executable or an inline shell script.
    /// Called with LLMUX_* env vars set to the model configuration.
    /// Exit 0 = healthy, non-zero = unhealthy.
    pub alive: String,
}

impl ModelConfig {
    fn is_valid_hostname(host: &str) -> bool {
        if host.parse::<IpAddr>().is_ok() {
            return true;
        }

        // "link-local IPv6 zone IDs are not supported". -- e.g. "fe80::1%eth0"
        if host.contains('%') {
            return false;
        }

        if host.is_empty() || host.len() > 253 {
            return false;
        }
        host.split('.').all(|label| {
            !label.is_empty()
                && label.len() <= 63
                && label.chars().all(|c| c.is_ascii_alphanumeric() || c == '-')
                && !label.starts_with('-')
                && !label.ends_with('-')
        })
    }

    /// validates syntax and bounds, but does not perform network lookups
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            ModelConfig::is_valid_hostname(&self.host),
            "invalid hostname: {:?}",
            self.host
        );

        // we'll be calling this at runtime when forwarding the connection.
        // better check now than fail later.
        let authority = if self.host.contains(":") {
            // ipv6 addresses require braces [] in URLs.
            format!("[{}]", self.host)
        } else {
            self.host.to_string()
        };
        format!("http://{}:{}/", authority, self.port).parse::<Uri>()?;

        if self.port == 0 {
            Err(anyhow!("port 0 is reserved"))
        } else {
            Ok(())
        }
    }
}

fn default_port() -> u16 {
    3000
}

fn default_host() -> String {
    "localhost".to_string()
}

impl Config {
    /// Load configuration from a YAML or JSON file (detected by extension).
    pub async fn from_file(path: &Path) -> Result<Self> {
        let contents = tokio::fs::read_to_string(path)
            .await
            .with_context(|| format!("Failed to read config file: {}", path.display()))?;

        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

        let raw_config: Config = match ext {
            "yaml" | "yml" => serde_yaml::from_str(&contents)
                .with_context(|| format!("Failed to parse YAML config: {}", path.display())),
            _ => serde_json::from_str(&contents)
                .with_context(|| format!("Failed to parse JSON config: {}", path.display())),
        }?;

        raw_config.validate()?;
        Ok(raw_config)
    }

    fn validate(&self) -> Result<()> {
        for (model_name, model_config) in &self.models {
            model_config
                .validate()
                .with_context(|| format!("model {:?} failed validation", model_name))?;
        }
        Ok(())
    }
}

/// Policy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyConfig {
    /// Request timeout in seconds. None = unlimited (requests wait forever).
    #[serde(default)]
    pub request_timeout_secs: Option<u64>,

    /// Whether to drain in-flight requests before switching
    #[serde(default = "default_drain_before_switch")]
    pub drain_before_switch: bool,

    /// Minimum seconds a model must stay active before it can be put to sleep.
    /// Prevents rapid wake/sleep thrashing. Default: 0 (no minimum).
    #[serde(default)]
    pub min_active_secs: u64,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            request_timeout_secs: None,
            drain_before_switch: default_drain_before_switch(),
            min_active_secs: 0,
        }
    }
}

fn default_drain_before_switch() -> bool {
    true
}

impl PolicyConfig {
    pub fn build_policy(&self) -> Box<dyn crate::policy::SwitchPolicy> {
        Box::new(crate::policy::FifoPolicy::new(
            self.request_timeout_secs.map(Duration::from_secs),
            self.drain_before_switch,
            Duration::from_secs(self.min_active_secs),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_invalid_port() {
        let mc = ModelConfig {
            host: "localhost".to_string(),
            port: 0,
            wake: "".to_string(),
            sleep: "".to_string(),
            alive: "".to_string(),
        };
        let err = mc.validate().expect_err("port 0 is bad");
        assert!(err.to_string().contains("port"));
    }

    #[test]
    fn test_bad_hosts_are_rejected() {
        let cases = [
            ("10.0.0.2", true),
            ("example.com", true),
            ("bad host", false),
            ("  ", false),
            ("localhost:5000", false),
            ("::1", true),
            ("foo-%2D-blah", false),
            ("http://localhost", false),
            (" fat-finger.com", false),
            ("10..54", false),
            ("", false),
        ];
        for (input, expected) in cases {
            let mc = ModelConfig {
                host: input.to_string(),
                port: 8000,
                wake: "".to_string(),
                sleep: "".to_string(),
                alive: "".to_string(),
            };
            assert_eq!(mc.validate().is_ok(), expected, "input: {input:?}");
        }
    }

    #[tokio::test]
    async fn test_parse_from_file_applies_validation() {
        let json_with_bad_host = r#"{
            "models": {
                "llama": {
                    "port": 8001,
                    "host": "-bad.host-",
                    "wake": "./scripts/wake-llama.sh",
                    "sleep": "./scripts/sleep-llama.sh",
                    "alive": "./scripts/alive-llama.sh"
                }
            },
            "policy": {
                "request_timeout_secs": 30
            },
            "port": 3000
        }"#;
        let mut file = tempfile::NamedTempFile::new().expect("should work");
        writeln!(file, "{json_with_bad_host}").expect("should be able to write");
        let path = file.path();
        let err = Config::from_file(path)
            .await
            .expect_err("supposed to crash on a bad host name");
        assert!(err.to_string().contains("model \"llama\" failed validation"), "{:?}", err);
    }

    #[test]
    fn test_parse_json() {
        let json = r#"{
            "models": {
                "llama": {
                    "port": 8001,
                    "host": "host.docker.internal",
                    "wake": "./scripts/wake-llama.sh",
                    "sleep": "./scripts/sleep-llama.sh",
                    "alive": "./scripts/alive-llama.sh"
                },
                "mistral": {
                    "port": 8002,
                    "wake": "./scripts/wake-mistral.sh",
                    "sleep": "./scripts/sleep-mistral.sh",
                    "alive": "./scripts/alive-mistral.sh"
                }
            },
            "policy": {
                "request_timeout_secs": 30
            },
            "port": 3000
        }"#;

        let config: Config = serde_json::from_str(json).unwrap();
        config.validate().expect("it's valid");
        assert_eq!(config.models.len(), 2);
        assert_eq!(config.models["llama"].port, 8001);
        assert_eq!(config.models["mistral"].host, "localhost");
        assert_eq!(config.models["llama"].host, "host.docker.internal");
        assert_eq!(config.policy.request_timeout_secs, Some(30));
    }

    #[test]
    fn test_parse_yaml() {
        let yaml = r#"
models:
  llama:
    port: 8001
    wake: ./scripts/wake-llama.sh
    sleep: |
      kill $(cat /tmp/llama.pid)
      rm /tmp/llama.pid
    alive: curl -sf http://localhost:8001/health
policy:
  request_timeout_secs: 60
port: 4000
"#;

        let config: Config = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.models.len(), 1);
        assert_eq!(config.models["llama"].port, 8001);
        assert_eq!(config.models["llama"].host, "localhost");
        assert_eq!(config.models["llama"].wake, "./scripts/wake-llama.sh");
        assert!(config.models["llama"].sleep.contains("kill"));
        assert_eq!(config.policy.request_timeout_secs, Some(60));
        assert_eq!(config.port, 4000);
    }

    #[test]
    fn test_defaults() {
        let json = r#"{
            "models": {
                "llama": {
                    "port": 8001,
                    "wake": "./wake.sh",
                    "sleep": "./sleep.sh",
                    "alive": "./alive.sh"
                }
            }
        }"#;

        let config: Config = serde_json::from_str(json).unwrap();
        assert_eq!(config.port, 3000);
        assert_eq!(config.policy.request_timeout_secs, None);
        assert!(config.policy.drain_before_switch);
        assert_eq!(config.policy.min_active_secs, 0);
    }
}

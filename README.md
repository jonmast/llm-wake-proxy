# llm-wake-proxy

> **⚠️ Alpha quality / personal project.** This works for my setup but has rough edges, minimal error handling in places, and no stability guarantees. Not recommended for production or any environment where reliability matters.

A standalone Rust proxy that keeps a private `llama.cpp` model host asleep when idle and wakes it on demand. Exposes an OpenAI-compatible API so standard clients like `opencode` work without custom glue.

## Architecture

```
Clients (opencode, etc.)
    |
    v
llm-wake-proxy (Kubernetes)
    |  - Request admission + queueing
    |  - Cold-start orchestration
    |  - SSH tunnel lifecycle
    |
    v
SSH Tunnel --> llama-server (loopback on bare-metal host)
```

The proxy owns the OpenAI-compatible API, cold-start orchestration, and SSH tunnel lifecycle. The bare-metal host runs `llama-server` and an inhibit holder under `systemd --user` only when needed.

## Configuration

All configuration is via environment variables.

### Proxy

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `3000` | Proxy listen port |
| `MODEL_ALIAS` | `llm-wake-proxy` | Stable model name exposed to clients |
| `MODEL_OWNED_BY` | `llm-wake-proxy` | Owner string for `/v1/models` |
| `MODEL_PROVIDER_ID` | `llama.cpp` | Provider ID for `/v1/models` |
| `EMBEDDINGS_ENABLED` | `true` | Enable embeddings forwarding |
| `WARM_MAX_ACTIVE_REQUESTS` | `2` | Max concurrent upstream requests |
| `WARM_MAX_QUEUED_REQUESTS` | `16` | Max queued warm requests (0 = no queue) |
| `WARM_QUEUE_TIMEOUT_SECS` | `30` | Max seconds a request waits in queue |
| `COLD_START_MAX_WAITING` | `32` | Max concurrent cold-start waiting requests |

### Host / SSH

| Variable | Default | Description |
|----------|---------|-------------|
| `SSH_HOST` | (required) | Bare-metal host address (IP or Tailscale hostname) |
| `SSH_USER` | (required) | SSH user on the host |
| `SSH_PORT` | `22` | SSH port |
| `HELPER_PATH` | `/usr/local/bin/helper` | Path to helper binary on host |
| `MODEL_PATH` | (required) | Path to model file on host (for model verification) |
| `TUNNEL_LOCAL_PORT` | `18080` | Local port for SSH tunnel |
| `LLAMA_SERVER_PORT` | `8080` | Remote port on host (llama-server) |

### Wake-on-LAN

| Variable | Default | Description |
|----------|---------|-------------|
| `WOL_MAC_ADDRESS` | (required) | MAC address of host (colon-separated) |
| `WOL_BROADCAST_ADDR` | `255.255.255.255` | Broadcast address |
| `WOL_PORT` | `9` | WOL UDP port |

## Host Setup

### Prerequisites

- `llama-server` installed and accessible on the host
- SSH key-based auth (no password) from the proxy's service account to the host
- Host SSH host key already accepted (`StrictHostKeyChecking=accept-new`)
- `systemd --user` managing `llama-server` and the inhibit holder

### Helper Binary

The helper binary (`llm-wake-proxy-helper`) runs on the host over SSH. It provides:

```bash
# Machine-readable host status
llm-wake-proxy-helper status

# Ensure llama-server is started (idempotent, singleton)
llm-wake-proxy-helper ensure-started --model-path /models/model.gguf --model-alias default

# Lease management
llm-wake-proxy-helper lease acquire --ttl 3600
llm-wake-proxy-helper lease release
llm-wake-proxy-helper lease inspect
```

All subcommands emit JSON on stdout and reserve stderr for human diagnostics.

### systemd Units

The helper manages these user-level systemd units:

- `llama-server.service` - The `llama-server` process
- `llm-inhibit-holder.service` - Transient systemd inhibitor to prevent suspend during lease

## Deployment (Kubernetes)

A multi-stage Dockerfile and Helm chart ship in this repo.

### Build and push the image

```bash
# One-time: log in to your registry (example uses GHCR).
echo $GITHUB_TOKEN | docker login ghcr.io -u <your-user> --password-stdin

# Build and push. Override IMAGE / TAG to taste.
make build IMAGE=ghcr.io/<your-user>/llm-wake-proxy TAG=0.1.0
make push  IMAGE=ghcr.io/<your-user>/llm-wake-proxy TAG=0.1.0

# Or multi-arch in one shot (requires buildx):
make buildx IMAGE=ghcr.io/<your-user>/llm-wake-proxy TAG=0.1.0
```

The image is a distroless `cc-debian12:nonroot` base with both `llm-wake-proxy` and `llm-wake-proxy-helper` plus the OpenSSH client (for the tunnel and helper RPC). It runs as UID 65532 with all capabilities dropped, a read-only root filesystem, and `seccompProfile: RuntimeDefault`.

> **Note:** The chart defaults to `hostNetwork: true` because Wake-on-LAN requires sending a UDP broadcast from the node’s physical network interface. Standard pod networking isolates the broadcast, so the magic packet never reaches the LAN.

### Provide the SSH key

The proxy opens an SSH tunnel to the bare-metal host, so the pod needs a private key. Create the Secret **out of band** so the key never lands in git:

```bash
ssh-keygen -t ed25519 -N '' -f ~/.ssh/llm-wake-proxy
ssh-keyscan -H llama.tailnet.example > known_hosts

kubectl create namespace llm-wake-proxy

kubectl create secret generic llm-wake-proxy-ssh-key \
  --namespace llm-wake-proxy \
  --from-file=ssh-privatekey=~/.ssh/llm-wake-proxy \
  --from-file=known_hosts=known_hosts
```

Authorise the public key on the bare-metal host as usual (e.g. `~/.ssh/authorized_keys`).

### Install the chart

```bash
helm install llm-wake-proxy ./charts/llm-wake-proxy \
  --namespace llm-wake-proxy \
  --set ssh.host=llama.tailnet.example \
  --set ssh.user=jon \
  --set wol.macAddress=AA:BB:CC:DD:EE:FF \
  --set ssh.modelPath=/models/qwen2.5-7b-instruct-q4_k_m.gguf \
  --set proxy.modelAlias=qwen2.5-7b-instruct \
  --set ssh.existingSecret=llm-wake-proxy-ssh-key
```

Required values: `ssh.host`, `ssh.user`, `ssh.modelPath`, `wol.macAddress`. The chart enforces these with `required` and `helm install` will refuse to proceed without them.

### Verify

```bash
kubectl --namespace llm-wake-proxy port-forward \
  svc/llm-wake-proxy 8080:3000

curl -s http://localhost:8080/healthz
# {"status":"ok"}

curl -s http://localhost:8080/status | jq
```

A cold host will return `503 warming_up` from `/v1/chat/completions` with a `Retry-After` header until WOL, SSH, and `llama-server` are all ready. See **Verification** below for the full status contract.

### Chart values

The full list lives in `charts/llm-wake-proxy/values.yaml`. Highlights:

| Value | Default | Notes |
|-------|---------|-------|
| `replicaCount` | `1` | V1 keeps coordination state in memory. No HA. |
| `service.type` | `ClusterIP` | Use `LoadBalancer`/`NodePort` to expose externally. |
| `resources.requests/limits` | 100m/128Mi → 1000m/512Mi | Tune for your model. |
| `probes.liveness/readiness` | enabled | Both hit `/healthz`. |
| `ssh.mountPath` | `/home/nonroot/.ssh` | Override if you ship a different layout. |
| `proxy.extraEnv` | `[]` | Merge arbitrary `env:` entries (e.g. `RUST_LOG=info`). |

### Other useful targets

```bash
make lint        # cargo check + clippy + helm lint
make render      # helm template dry-run with sane defaults
make package     # helm package the chart into dist/
make uninstall   # helm uninstall llm-wake-proxy
```

## Verification

### Health

```bash
curl http://localhost:8080/healthz
# {"status":"ok"}
```

### Status

```bash
curl http://localhost:8080/status
```

Returns:
```json
{
  "state": "ready",
  "model_alias": "default",
  "capabilities": {
    "chat": "ready",
    "embeddings": "ready"
  },
  "tunnel": "ready",
  "last_wake_attempt_at": 1717156800,
  "lease_expires_at": 1717158600,
  "host_unit": {
    "llama_server_unit": "active",
    "inhibit_unit": "activating"
  },
  "metrics": {
    "cold_starts": 1,
    "warm_requests": 42,
    "queue_full_rejections": 0,
    "queue_timeouts": 0,
    "wake_attempts": 1,
    "wake_failures": 0,
    "tunnel_drops": 0,
    "embeddings_degraded": 0,
    "forwarding_errors": 0,
    "chat_requests": 42,
    "embeddings_requests": 5
  }
}
```

### State Transitions

| State | Meaning |
|-------|---------|
| `cold` | Backend has never been probed or needs a fresh wake |
| `warming` | Wake sent, waiting for SSH + helper |
| `ready` | Backend is live and tunnel is established |
| `error` | Something failed (SSH, wake, helper, tunnel) |

### Error Semantics

| Status | Type | Meaning |
|--------|------|---------|
| `503` | `warming_up` | Backend is starting; retry after `Retry-After` |
| `503` | `backend_error` | Backend observation failed |
| `503` | `backend_unavailable` | Backend ready but forwarding failed |
| `429` | `overloaded` | Warm execution queue is full or timed out |
| `400` | `invalid_request_error` | Bad JSON, unsupported fields, bad model |
| `400` | `unsupported_embeddings` | Embeddings disabled or degraded |

## Metrics

The `/status` endpoint includes a `metrics` object with atomic counters:

- `cold_starts` - Number of cold-start transitions
- `warm_requests` - Requests served via warm path
- `queue_full_rejections` - Requests rejected (queue full)
- `queue_timeouts` - Requests that timed out in queue
- `wake_attempts` - WOL packets sent
- `wake_failures` - WOL/SSH failures during wake
- `tunnel_drops` - SSH tunnel disconnections
- `embeddings_degraded` - Embeddings degraded transitions
- `forwarding_errors` - Upstream forwarding errors
- `chat_requests` - Total chat requests received
- `embeddings_requests` - Total embeddings requests received

## Lifecycle Timing

| Variable | Default | Description |
|----------|---------|-------------|
| `COLD_WAIT_BUDGET_SECS` | `90` | How long a cold request waits before returning 503 |
| `HARD_BOOT_DEADLINE_SECS` | `300` | Maximum time from first wake to backend ready |
| `BOOTSTRAP_POLL_INTERVAL_MS` | `1000` | Polling interval during bootstrap |
| `RETRY_AFTER_SECS` | `10` | Value for Retry-After header in warming responses |

## Design Constraints

- **Single replica**: V1 keeps coordination state in memory. No HA.
- **No auth**: The proxy accepts `Authorization` headers but does not enforce auth. Use network-level access control.
- **Private LAN only**: Not intended for public internet exposure.
- **No root/sudo**: The service and helper run as unprivileged user.
- **Host stays loopback-only**: All inference traffic routes through the SSH tunnel.

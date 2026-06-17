# Context: llm-wake-proxy

## Glossary

### Host-side units

**`llama-server.service`**
Static `systemd --user` unit file. Shipped in this repo at the repo root. Install by copying to `~/.config/systemd/user/` on the host and editing `ExecStart` to match your model path. Runs `llama-server` on `127.0.0.1:8080`. Kept **disabled** so it only starts on demand via the helper. The host can sleep when idle because this unit is not running.

**`llama-server-embeddings.service`**
Optional second static `systemd --user` unit file, also shipped at the repo root, for **dual-backend mode** (separate chat vs embeddings models). Same shape as `llama-server.service` but runs a dedicated embeddings model on `127.0.0.1:8081` with `--embedding`. Only relevant on hosts where the proxy is configured with `EMBEDDINGS_MODEL_PATH`. Kept **disabled** like the chat unit. The helper looks for it under the unit name `llama-server-embeddings` and port `8081` by default, overridable via `LLAMA_SERVER_EMBEDDINGS_UNIT` / `LLAMA_SERVER_EMBEDDINGS_PORT`.

**`llm-wake-proxy-inhibit`**
Transient `systemd --user` unit created by `systemd-run` when the proxy acquires a lease. Runs `systemd-inhibit --what=sleep` to block the desktop environment's auto-suspend while requests are active. Removed when the lease is released or expires. Not a static `.service` file — it is ephemeral by design.

### Helper binary

**`llm-wake-proxy-helper`**
Runs on the bare-metal host over SSH. Provides `status`, `ensure-started`, and `lease` subcommands. Installed manually by copying the Cargo-built binary (default path: `/usr/local/bin/llm-wake-proxy-helper`). No special permissions required for `systemd-inhibit` on most distros, though polkit configuration may be needed on some systems.

### Proxy invocation

The proxy calls the helper over SSH as:
```
ssh -i <key> user@host env EXPECTED_MODEL_PATH=<path> /usr/local/bin/llm-wake-proxy-helper ensure-started <chat|embeddings> <alias>
```
The `EXPECTED_MODEL_PATH` is passed as an env var, not a CLI flag. The `<chat|embeddings>` target is a required positional argument selecting which `llama-server` unit/port the helper manages (`llama-server`/`8080` for `chat`, `llama-server-embeddings`/`8081` for `embeddings`, overridable via `LLAMA_SERVER_EMBEDDINGS_UNIT`/`LLAMA_SERVER_EMBEDDINGS_PORT`).

In **dual-backend mode** (`EMBEDDINGS_MODEL_PATH` set on the proxy), the proxy runs two independent lifecycle managers, each issuing its own `ensure-started` call — one with `chat <chat-alias>` against the chat model path/port, one with `embeddings <embeddings-alias>` against the embeddings model path/port — over separate SSH tunnels. In the default shared-backend configuration, only the `chat` target is ever used and both `/v1/chat/completions` and `/v1/embeddings` are forwarded to the same `llama-server` process.

### Sleep behavior

The host uses desktop-environment auto-suspend (GNOME/KDE power settings). The `systemd-inhibit` from the lease holder prevents suspend during active sessions. When the lease expires, the inhibitor is removed and the desktop environment can suspend again.

The SSH tunnel may or may not survive a suspend cycle — this is not guaranteed and should not be relied upon. The proxy will cold-start the host on the next request if the tunnel is broken.

### Linger

`loginctl enable-linger` is required if the host needs to work while nobody is logged in (i.e., the proxy needs to SSH in and start llama-server without an active desktop session).

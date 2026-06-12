# Context: llm-wake-proxy

## Glossary

### Host-side units

**`llama-server.service`**
Static `systemd --user` unit file. Shipped in this repo at the repo root. Install by copying to `~/.config/systemd/user/` on the host and editing `ExecStart` to match your model path. Runs `llama-server` on `127.0.0.1:8080`. Kept **disabled** so it only starts on demand via the helper. The host can sleep when idle because this unit is not running.

**`llm-wake-proxy-inhibit`**
Transient `systemd --user` unit created by `systemd-run` when the proxy acquires a lease. Runs `systemd-inhibit --what=sleep` to block the desktop environment's auto-suspend while requests are active. Removed when the lease is released or expires. Not a static `.service` file — it is ephemeral by design.

### Helper binary

**`llm-wake-proxy-helper`**
Runs on the bare-metal host over SSH. Provides `status`, `ensure-started`, and `lease` subcommands. Installed manually by copying the Cargo-built binary (default path: `/usr/local/bin/llm-wake-proxy-helper`). No special permissions required for `systemd-inhibit` on most distros, though polkit configuration may be needed on some systems.

### Proxy invocation

The proxy calls the helper over SSH as:
```
ssh -i <key> user@host env EXPECTED_MODEL_PATH=<path> /usr/local/bin/llm-wake-proxy-helper ensure-started <alias>
```
The `EXPECTED_MODEL_PATH` is passed as an env var, not a CLI flag.

### Sleep behavior

The host uses desktop-environment auto-suspend (GNOME/KDE power settings). The `systemd-inhibit` from the lease holder prevents suspend during active sessions. When the lease expires, the inhibitor is removed and the desktop environment can suspend again.

The SSH tunnel may or may not survive a suspend cycle — this is not guaranteed and should not be relied upon. The proxy will cold-start the host on the next request if the tunnel is broken.

### Linger

`loginctl enable-linger` is required if the host needs to work while nobody is logged in (i.e., the proxy needs to SSH in and start llama-server without an active desktop session).

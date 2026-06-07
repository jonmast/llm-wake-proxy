# syntax=docker/dockerfile:1.7

# Multi-stage build for llm-wake-proxy.
#
# Produces a distroless image containing:
#   - /usr/local/bin/llm-wake-proxy           (the proxy)
#   - /usr/local/bin/llm-wake-proxy-helper    (the host-side helper)
#   - /usr/bin/ssh                            (for SSH tunnel + helper RPC)
#
# Build:    docker build -t ghcr.io/<user>/llm-wake-proxy:dev .
# Push:     docker push ghcr.io/<user>/llm-wake-proxy:dev
# Inspect:  docker run --rm ghcr.io/<user>/llm-wake-proxy:dev --version
#
# Notes:
#   - Uses buildkit cache mounts for cargo registry/git/target. No third-party
#     base image; only docker.io/library/* and gcr.io/distroless/* are pulled.
#   - Final image is distroless cc-debian12:nonroot (UID 65532).

ARG RUST_VERSION=1.88
ARG DEBIAN_RELEASE=bookworm

# ---- build ------------------------------------------------------------------
FROM docker.io/library/rust:${RUST_VERSION}-${DEBIAN_RELEASE} AS builder
WORKDIR /app
ENV CARGO_TERM_COLOR=always
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        openssh-client \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock ./
COPY src ./src

# Cache mounts give proper layer-level caching for cargo registry/git/target
# without needing a separate dependency-recipe stage.
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/usr/local/cargo/git,sharing=locked \
    --mount=type=cache,target=/app/target,sharing=locked \
    cargo build --release --bins \
    && install -m 0755 target/release/llm-wake-proxy         /usr/local/bin/llm-wake-proxy \
    && install -m 0755 target/release/helper                  /usr/local/bin/llm-wake-proxy-helper \
    && strip /usr/local/bin/llm-wake-proxy /usr/local/bin/llm-wake-proxy-helper

# ---- ssh runtime payload (slim extract) -------------------------------------
FROM docker.io/library/debian:${DEBIAN_RELEASE}-slim AS ssh-runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends openssh-client \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /out \
    && cp /usr/bin/ssh /out/ssh \
    && cp -r /usr/lib/openssh /out/openssh \
    && cp -r /lib/x86_64-linux-gnu /out/lib-x86_64-linux-gnu

# ---- final distroless image -------------------------------------------------
FROM gcr.io/distroless/cc-debian12:nonroot

COPY --from=builder     /usr/local/bin/llm-wake-proxy          /usr/local/bin/llm-wake-proxy
COPY --from=builder     /usr/local/bin/llm-wake-proxy-helper   /usr/local/bin/llm-wake-proxy-helper
COPY --from=ssh-runtime /out/ssh                               /usr/bin/ssh
COPY --from=ssh-runtime /out/openssh                           /usr/lib/openssh
# ssh needs libselinux/libkrb5/libfido2/libbsd/libmd which aren't in
# distroless. Overlay them; the existing libs in cc-debian12 stay in place.
COPY --from=ssh-runtime /out/lib-x86_64-linux-gnu/             /lib/x86_64-linux-gnu/

# 65532 is the distroless "nonroot" UID.
USER 65532:65532
EXPOSE 3000

ENTRYPOINT ["/usr/local/bin/llm-wake-proxy"]

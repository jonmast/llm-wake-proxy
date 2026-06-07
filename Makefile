# llm-wake-proxy build / deploy helpers.
#
# Common usage:
#   make build                        # docker build, tag :dev
#   make push                         # docker push IMAGE=:dev
#   make lint                         # cargo check + cargo clippy + helm lint
#   make render                       # helm template dry-run to stdout
#   make install                      # helm install (requires --set overrides)
#   make package                      # helm package the chart into dist/

REGISTRY   ?= ghcr.io
IMAGE_USER ?= jonmast
IMAGE_NAME ?= llm-wake-proxy
IMAGE      ?= $(REGISTRY)/$(IMAGE_USER)/$(IMAGE_NAME)
TAG        ?= dev
PLATFORM   ?= linux/amd64
CHART      := charts/$(IMAGE_NAME)
DIST       := dist

CARGO     ?= cargo
HELM      ?= helm
DOCKER    ?= docker
BUILDX    ?= docker buildx

# Verbosity for cargo
CARGO_FLAGS ?=

# --- Docker ------------------------------------------------------------------

.PHONY: build
build: ## docker build (single-arch)
	$(DOCKER) build \
		--platform $(PLATFORM) \
		--tag $(IMAGE):$(TAG) \
		--tag $(IMAGE):latest \
		.

.PHONY: buildx
buildx: ## docker buildx build (multi-arch via buildx)
	$(BUILDX) build \
		--platform linux/amd64,linux/arm64 \
		--tag $(IMAGE):$(TAG) \
		--tag $(IMAGE):latest \
		--push \
		.

.PHONY: push
push: ## docker push $(TAG) and :latest
	$(DOCKER) push $(IMAGE):$(TAG)
	$(DOCKER) push $(IMAGE):latest

# --- Rust --------------------------------------------------------------------

.PHONY: check
check: ## cargo check
	$(CARGO) check $(CARGO_FLAGS) --all-targets

.PHONY: clippy
clippy: ## cargo clippy (deny warnings)
	$(CARGO) clippy $(CARGO_FLAGS) --all-targets -- -D warnings

.PHONY: test
test: ## cargo test
	$(CARGO) test $(CARGO_FLAGS) --all-targets

# --- Helm --------------------------------------------------------------------

.PHONY: helm-lint
helm-lint: ## helm lint the chart
	$(HELM) lint $(CHART)

.PHONY: render
render: ## helm template render (no install). Use --set to fill required values.
	$(HELM) template release $(CHART) \
		--set ssh.host=llama.example \
		--set ssh.user=jon \
		--set wol.macAddress=AA:BB:CC:DD:EE:FF \
		--set ssh.modelPath=/models/qwen2.5-7b-instruct-q4_k_m.gguf \
		--set proxy.modelAlias=qwen2.5-7b-instruct

.PHONY: install
install: ## helm install. Pass extra flags via HELM_FLAGS=...
	$(HELM) install llm-wake-proxy $(CHART) $(HELM_FLAGS)

.PHONY: upgrade
upgrade: ## helm upgrade. Pass extra flags via HELM_FLAGS=...
	$(HELM) upgrade llm-wake-proxy $(CHART) $(HELM_FLAGS)

.PHONY: uninstall
uninstall: ## helm uninstall
	$(HELM) uninstall llm-wake-proxy

.PHONY: package
package: ## helm package the chart into dist/
	@mkdir -p $(DIST)
	$(HELM) package $(CHART) --destination $(DIST)

# --- Meta --------------------------------------------------------------------

.PHONY: lint
lint: check clippy helm-lint ## run every static check

.PHONY: help
help: ## show this help
	@awk 'BEGIN {FS = ":.*##"; printf "Targets:\n"} \
		/^[a-zA-Z_-]+:.*##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 }' \
		$(MAKEFILE_LIST)

.DEFAULT_GOAL := help

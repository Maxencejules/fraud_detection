# Developer entry points. Windows users without `make` can run the commands directly.
.DEFAULT_GOAL := help
COMPOSE ?= docker compose

.PHONY: help
help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# --- local development -------------------------------------------------------------

.PHONY: install
install: ## Install all dependencies and git hooks
	uv sync --all-extras
	uvx pre-commit install

.PHONY: lint
lint: ## Lint and check formatting
	uv run ruff check .
	uv run ruff format --check .

.PHONY: format
format: ## Auto-fix lint issues and format code
	uv run ruff check --fix .
	uv run ruff format .

.PHONY: typecheck
typecheck: ## Static type checking (mypy --strict)
	uv run mypy

.PHONY: test
test: ## Unit tests with the coverage gate
	uv run pytest --cov

.PHONY: audit
audit: ## Scan locked dependencies for known vulnerabilities
	uv audit --preview-features audit-command --locked --ignore GHSA-8mgp-746c-j5xp

.PHONY: check
check: lint typecheck test ## Everything CI checks before the Docker stage

# --- Docker stack ------------------------------------------------------------------

.PHONY: up
up: ## Build and start the pipeline (bootstrap and training run automatically)
	$(COMPOSE) up -d --build

.PHONY: up-observability
up-observability: ## Start the pipeline plus Prometheus and Grafana
	$(COMPOSE) --profile observability up -d --build

.PHONY: ps
ps: ## Show service status
	$(COMPOSE) ps -a

.PHONY: logs
logs: ## Follow service logs
	$(COMPOSE) logs -f --tail=100

.PHONY: bootstrap
bootstrap: ## Re-simulate history and rebuild the online feature store
	$(COMPOSE) run --rm bootstrap python -m fraud_detection.bootstrap --reset

.PHONY: train
train: ## Train a challenger and promote it if it beats the champion
	$(COMPOSE) run --rm trainer python -m fraud_detection.trainer

.PHONY: smoke
smoke: ## End-to-end smoke test against the running stack
	python3 scripts/smoke_test.py

.PHONY: test-integration
test-integration: ## Integration tests against the running stack
	REQUIRE_STACK=1 uv run pytest tests/integration -m integration

.PHONY: benchmark
benchmark: ## Latency/throughput benchmark (writes benchmark.json)
	uv run python scripts/benchmark.py --json-out benchmark.json

.PHONY: down
down: ## Stop the stack (keeps data volumes)
	$(COMPOSE) --profile observability --profile tools down

.PHONY: clean
clean: ## Stop the stack and delete all data volumes
	$(COMPOSE) --profile observability --profile tools down -v --remove-orphans

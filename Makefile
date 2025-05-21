.PHONY: help install-deps tox

install-uv: ## Install required utilities/tools
	@command -v uv > /dev/null || { echo >&2 "uv is not installed. Installing..."; pip install --upgrade pip uv; }

install-global: install-uv uv-lock-check ## Install rca-accelerator-chatbot to global Python directories
	uv pip install .

install-deps: install-uv ## Install Python dependencies
	uv sync

install-dev-deps: install-uv ## Install Python dependencies
	uv sync --dev

uv-lock-check: ## Check that the uv.lock file is in a good shape
	uv lock --check

tox: ## Run tox
	tox

help: ## Show this help screen
	@echo 'Usage: make <OPTIONS> ... <TARGETS>'
	@echo ''
	@echo 'Available targets are:'
	@echo ''
	@grep -E '^[ a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'
	@echo ''

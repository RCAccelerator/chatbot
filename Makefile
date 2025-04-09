.PHONY: help install-deps install-pre-commit pre-commit-check install-dev-deps tox docs

install-pdm: ## Install required utilities/tools
	@command -v pdm > /dev/null || { echo >&2 "pdm is not installed. Installing..."; pip install --upgrade pip pdm; }

install-pre-commit: ## Install pre-commit if not already installed
	@command -v pre-commit > /dev/null || { echo >&2 "pre-commit is not installed. Installing..."; pip install pre-commit; }

install-global: install-pdm pdm-lock-check ## Install rca-accelerator-chatbot to global Python directories
	pdm install --global --project .

install-deps: install-pdm ## Install Python dependencies
	pdm sync

install-dev-deps: install-pdm install-pre-commit ## Install dev dependencies including pre-commit
	pdm sync --dev

pdm-lock-check: ## Check that the pdm.lock file is in a good shape
	pdm lock --check

pre-commit-check: install-pre-commit ## Run pre-commit check on all files
	pre-commit run --show-diff-on-failure --color=always --all-files

tox: ## Run tox
	tox

lint: tox pre-commit-check ## Run tox and pre-commit checks

docs: ## Generate documentation by running docs/prepare_env.sh and docs/build.sh
	cd docs; ./prepare_env.sh && ./build.sh

help: ## Show this help screen
	@echo 'Usage: make <OPTIONS> ... <TARGETS>'
	@echo ''
	@echo 'Available targets are:'
	@echo ''
	@grep -E '^[ a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'
	@echo ''

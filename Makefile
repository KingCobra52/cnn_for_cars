# carvision — one command per pipeline stage.
# Everything except `setup` assumes the venv is active.

PY      ?= python
VENV    ?= .venv
BACKBONE ?= dinov2_vits14
HEAD     ?= linear
SEED     ?= 0

.DEFAULT_GOAL := help

.PHONY: help
help:  ## Show this help.
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ----------------------------------------------------------------- setup

.PHONY: setup
setup:  ## Create a venv and install the package with dev extras.
	$(PY) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e ".[dev,export,app]"
	$(VENV)/bin/pre-commit install

.PHONY: lock
lock:  ## Regenerate requirements.lock from the current environment.
	$(VENV)/bin/pip freeze --exclude-editable > requirements.lock

# ----------------------------------------------------------------- quality

.PHONY: lint
lint:  ## Run ruff (lint + format check).
	ruff check src tests app
	ruff format --check src tests app

.PHONY: format
format:  ## Autoformat and autofix.
	ruff check --fix src tests app
	ruff format src tests app

.PHONY: typecheck
typecheck:  ## Run mypy in strict mode over src/.
	mypy

.PHONY: test
test:  ## Run the test suite (no network, no dataset).
	pytest -m "not slow"

.PHONY: check
check: lint typecheck test  ## Everything CI runs.

# ----------------------------------------------------------------- pipeline

.PHONY: data
data:  ## Download Stanford Cars and write the deterministic splits.
	carvision data download
	carvision data split

.PHONY: cache
cache:  ## Build the frozen-backbone embedding cache for one backbone.
	carvision cache build --backbone $(BACKBONE)

.PHONY: cache-all
cache-all:  ## Build the embedding cache for every backbone (slow, once).
	for b in resnet50 clip_vitb32 dinov2_vits14; do carvision cache build --backbone $$b; done

.PHONY: train
train:  ## Train one head on cached embeddings.
	carvision train backbone=$(BACKBONE) head=$(HEAD) seed=$(SEED)

.PHONY: sweep
sweep:  ## 3 backbones x 2 heads x 5 seeds on cached embeddings.
	carvision sweep

.PHONY: zeroshot
zeroshot:  ## CLIP zero-shot baseline (no training).
	carvision zeroshot

.PHONY: eval
eval:  ## Evaluate the best run: metrics, CIs, calibration.
	carvision eval --run best

.PHONY: figures
figures:  ## Regenerate every figure in docs/figures/.
	carvision figures

.PHONY: report
report:  ## Regenerate docs/RESULTS.md from the evaluated runs.
	carvision report

.PHONY: export
export:  ## Export to ONNX and verify parity with PyTorch.
	carvision export --run best

.PHONY: bench
bench:  ## Benchmark CPU latency, PyTorch vs ONNX Runtime.
	carvision bench

.PHONY: demo
demo:  ## Run the Gradio demo locally.
	cd app && $(PY) app.py

.PHONY: all
all: data cache-all sweep zeroshot eval figures report export bench  ## Full pipeline from scratch.

# ----------------------------------------------------------------- misc

.PHONY: clean
clean:  ## Remove caches and build artifacts (keeps data/ and artifacts/).
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml build dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

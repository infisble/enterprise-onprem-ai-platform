# NovaPromo enterprise AI platform - common tasks.
# Windows users without make: see the equivalent commands in README.md.
PY ?= python
VENV ?= .venv
BIN := $(VENV)/bin
ifeq ($(OS),Windows_NT)
  BIN := $(VENV)/Scripts
endif
API ?= http://localhost:$${API_PORT:-8000}

.PHONY: help setup data bootstrap run web demo test lint format typecheck check eval diagrams \
        up down logs demo-docker test-docker clean

help:            ## Show targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

setup:           ## Create venv + install Python and web dependencies
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -e ".[dev]"
	cd apps/web && npm install

data:            ## Regenerate synthetic supplier feeds and documents (already committed)
	$(BIN)/python scripts/generate_synthetic_data.py
	$(BIN)/python scripts/generate_documents.py

bootstrap:       ## Local (no Docker) bootstrap: SQLite + embedded Qdrant + BM25
	$(BIN)/python scripts/bootstrap_demo.py --reset

run:             ## Run the API locally with hot reload (port 8000)
	$(BIN)/uvicorn apps.api.main:app --reload --port 8000

web:             ## Run the web UI dev server (port 5174, proxies to the API)
	cd apps/web && VITE_API_PROXY=http://127.0.0.1:8000 npm run dev

demo:            ## Run all demo scenarios in-process and print the report
	$(BIN)/python scripts/run_demo_scenarios.py

test:            ## Run the test suite
	$(BIN)/python -m pytest

lint:            ## Ruff lint
	$(BIN)/ruff check .

format:          ## Ruff format
	$(BIN)/ruff format .

typecheck:       ## mypy + TypeScript
	$(BIN)/mypy apps services packages
	cd apps/web && npm run typecheck

check: lint typecheck test  ## Everything CI runs

eval:            ## Run the model evaluation harness (writes evaluations/reports/)
	$(BIN)/python -m evaluations.run_eval

diagrams:        ## Render Mermaid diagrams to SVG/PNG (needs Node)
	$(BIN)/python scripts/render_diagrams.py

up:              ## docker compose up (core stack)
	docker compose up --build -d

down:            ## Stop the stack
	docker compose down

logs:            ## Tail API + worker logs
	docker compose logs -f api worker

demo-docker:     ## Run demo scenarios against the Docker API
	$(BIN)/python scripts/run_demo_scenarios.py --api $(API)

test-docker:     ## Run the test suite inside the API container (isolated SQLite + embedded stores)
	docker compose exec api python -m pytest -q

clean:           ## Remove local data and caches
	rm -rf .data .pytest_cache .mypy_cache .ruff_cache

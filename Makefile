.PHONY: setup data test lint format reproduce clean

setup:  ## Install the package with dev dependencies
	pip install -e ".[dev]"

data:  ## Download the Coat dataset (falls back to synthetic)
	bcr-download --data-dir data/raw

test:  ## Run the test suite
	pytest -q

lint:  ## Lint with ruff
	ruff check src tests
	ruff format --check src tests

format:  ## Auto-format with ruff
	ruff format src tests
	ruff check --fix src tests

reproduce:  ## Regenerate all reported results from seed 42 (~30-45 min on CPU)
	python experiments/run_all.py

clean:  ## Remove caches and build artifacts
	rm -rf .pytest_cache **/__pycache__ build *.egg-info src/*.egg-info

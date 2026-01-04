# LIBRA Makefile
# Common development commands

.PHONY: help install dev lint format type-check test test-cov bench clean pre-commit

# Default target
help:
	@echo "LIBRA Development Commands"
	@echo "=========================="
	@echo ""
	@echo "Setup:"
	@echo "  make install      Install production dependencies"
	@echo "  make dev          Install development dependencies"
	@echo "  make pre-commit   Install pre-commit hooks"
	@echo ""
	@echo "Quality:"
	@echo "  make lint         Run ruff linter"
	@echo "  make format       Format code with ruff"
	@echo "  make type-check   Run mypy type checker"
	@echo "  make check        Run all checks (lint + type-check)"
	@echo ""
	@echo "Testing:"
	@echo "  make test         Run tests"
	@echo "  make test-cov     Run tests with coverage"
	@echo "  make test-fast    Run tests in parallel"
	@echo ""
	@echo "Benchmarks:"
	@echo "  make bench            Run all benchmarks"
	@echo "  make bench-quick      Quick summary benchmark"
	@echo "  make bench-throughput Throughput benchmarks (pytest-benchmark)"
	@echo "  make bench-latency    Latency distribution (HDR Histogram)"
	@echo "  make bench-e2e        End-to-end benchmarks"
	@echo "  make bench-save       Save benchmark results to JSON"
	@echo "  make bench-compare    Compare with saved results"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean        Remove build artifacts"
	@echo "  make update       Update dependencies"

# =============================================================================
# Setup
# =============================================================================

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

pre-commit:
	pre-commit install
	pre-commit install --hook-type commit-msg
	@echo "Pre-commit hooks installed!"

# =============================================================================
# Quality Checks
# =============================================================================

lint:
	ruff check src/ tests/

lint-fix:
	ruff check --fix src/ tests/

format:
	ruff format src/ tests/

format-check:
	ruff format --check src/ tests/

type-check:
	mypy src/

check: lint-fix format type-check
	@echo "All checks passed!"

# =============================================================================
# Testing
# =============================================================================

test:
	pytest tests/

test-cov:
	pytest tests/ --cov=libra --cov-report=term-missing --cov-report=html

test-fast:
	pytest tests/ -n auto

test-unit:
	pytest tests/unit/

test-integration:
	pytest tests/integration/

bench:
	pytest benchmarks/ --benchmark-only -v

bench-throughput:
	pytest benchmarks/bench_publish.py -v -m bench_throughput --benchmark-only

bench-latency:
	pytest benchmarks/bench_latency.py -v -s

bench-e2e:
	pytest benchmarks/bench_full_cycle.py -v -s

bench-save:
	pytest benchmarks/ --benchmark-only --benchmark-autosave --benchmark-json=.benchmarks/latest.json

bench-compare:
	pytest benchmarks/ --benchmark-only --benchmark-compare

bench-quick:
	pytest benchmarks/bench_full_cycle.py::TestBenchmarkSummary -v -s

# =============================================================================
# Maintenance
# =============================================================================

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf src/*.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

update:
	pip install --upgrade pip
	pip install -e ".[dev]" --upgrade
	pre-commit autoupdate

# =============================================================================
# CI/CD helpers
# =============================================================================

ci-check:
	ruff check src/ tests/
	ruff format --check src/ tests/
	mypy src/
	pytest tests/ --cov=libra --cov-fail-under=80

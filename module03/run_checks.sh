#!/usr/bin/env bash
# Run from anywhere after activating the project's virtual environment.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! python -c 'import pytest' >/dev/null 2>&1; then
    echo "pytest is not installed. Run: python -m pip install pytest" >&2
    exit 2
fi

echo '==> Syntax checks'
python -m py_compile \
    module03/pipeline.py module03/reliability.py \
    module03/verify_artifacts.py module03/failure_lab.py \
    module03/tests/test_reliability.py module03/tests/test_integration.py

echo '==> Python dependency consistency'
python -m pip check

echo '==> Unit/reliability tests (offline)'
python -m pytest -v module03/tests/test_reliability.py

if [[ "${1:-}" == "--live" ]]; then
    echo '==> Read-only integration tests (requires running MLflow + Prefect)'
    MLOPS_INTEGRATION=1 python -m pytest -v module03/tests/test_integration.py
    echo '==> Inspecting actual stored MLflow artifacts'
    python module03/verify_artifacts.py
elif [[ $# -gt 0 ]]; then
    echo 'Usage: bash module03/run_checks.sh [--live]' >&2
    exit 2
else
    echo 'Live checks skipped. To enable: bash module03/run_checks.sh --live'
fi

echo '==> Checks complete'

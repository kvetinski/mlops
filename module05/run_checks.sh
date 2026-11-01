#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="$ROOT/.venv/bin/python"
"$PYTHON" -m pip check
"$PYTHON" -m py_compile module05/src/prepare_data.py module05/src/monitor.py \
  module05/src/collect_metrics.py module05/src/collect_drift.py \
  module05/src/check_monitoring.py
"$PYTHON" -m pytest -q module05/tests/test_monitoring_checks.py
"$PYTHON" module05/src/check_monitoring.py

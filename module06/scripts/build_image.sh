#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODEL=module04/homework/model.bin
CHECKSUM=module06/homework/model.sha256
if [[ ! -s "$MODEL" || ! -s "$CHECKSUM" ]]; then
  echo "Missing trusted model or model checksum; see module06/README.md" >&2
  exit 1
fi
sha256sum --check "$CHECKSUM"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/module04/homework" "$TMP/module06/homework"
cp module04/homework/Pipfile module04/homework/Pipfile.lock "$TMP/module04/homework/"
cp "$MODEL" "$TMP/module04/homework/model.bin"
cp module06/homework/batch.py "$TMP/module06/homework/batch.py"

# Minimal Docker build context: no large taxi datasets or virtual environments.
docker build \
  --network="${DOCKER_BUILD_NETWORK:-default}" \
  --file "$ROOT/module06/Dockerfile.batch" \
  --tag "${DOCKER_IMAGE:-taxi-batch:module06}" \
  "$TMP"

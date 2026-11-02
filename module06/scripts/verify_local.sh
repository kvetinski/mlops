#!/usr/bin/env bash
# Uses LocalStack on 127.0.0.1:4566; never connects to an AWS account.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export PIPENV_IGNORE_VIRTUALENVS=1
export PIPENV_PIPFILE="$ROOT/module04/homework/Pipfile"
export S3_ENDPOINT_URL="http://127.0.0.1:4566"
export S3_BUCKET="nyc-duration-iac"

make -f module06/Makefile check
make -f module06/Makefile tf-validate
curl --fail --silent --show-error --noproxy '*' \
  "$S3_ENDPOINT_URL/_localstack/health" > /dev/null
make -f module06/Makefile tf-apply
make -f module06/Makefile integration
make -f module06/Makefile image-build
make -f module06/Makefile image-smoke
printf '\nALL LOCAL MODULE 6 CHECKS PASSED\n'

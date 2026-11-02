# All targets are invoked from repository root:
# make -f module06/Makefile check
SHELL := /bin/bash
ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/..)
export PIPENV_PIPFILE := $(ROOT)/module04/homework/Pipfile
export PIPENV_IGNORE_VIRTUALENVS := 1
TARGET := module06/homework
TF_DIR := module06/terraform

.PHONY: format lint format-check test integration check hooks-install hooks-check tf-fmt tf-init tf-validate tf-plan tf-apply tf-destroy image-build image-smoke

format:
	cd "$(ROOT)" && pipenv run ruff check --fix --config module06/pyproject.toml $(TARGET)
	cd "$(ROOT)" && pipenv run ruff format --config module06/pyproject.toml $(TARGET)

lint:
	cd "$(ROOT)" && pipenv run ruff check --config module06/pyproject.toml $(TARGET)

format-check:
	cd "$(ROOT)" && pipenv run ruff format --check --config module06/pyproject.toml $(TARGET)

test:
	cd "$(ROOT)" && pipenv run python -m pytest -q module06/homework/tests

integration:
	cd "$(ROOT)" && S3_ENDPOINT_URL="$${S3_ENDPOINT_URL:-http://127.0.0.1:4566}" pipenv run python module06/homework/integration_test.py

check: lint format-check test

hooks-install:
	cd "$(ROOT)" && pipenv run pre-commit install --config module06/.pre-commit-config.yaml

hooks-check:
	cd "$(ROOT)" && pipenv run pre-commit run --config module06/.pre-commit-config.yaml --all-files

tf-fmt:
	cd "$(ROOT)" && terraform -chdir=$(TF_DIR) fmt -check -recursive

tf-init:
	cd "$(ROOT)" && terraform -chdir=$(TF_DIR) init -input=false

tf-validate: tf-init tf-fmt
	cd "$(ROOT)" && terraform -chdir=$(TF_DIR) validate

tf-plan: tf-validate
	cd "$(ROOT)" && terraform -chdir=$(TF_DIR) plan -input=false

tf-apply: tf-validate
	cd "$(ROOT)" && terraform -chdir=$(TF_DIR) apply -input=false -auto-approve

tf-destroy:
	cd "$(ROOT)" && terraform -chdir=$(TF_DIR) destroy -input=false -auto-approve

image-build:
	cd "$(ROOT)" && bash module06/scripts/build_image.sh

image-smoke:
	docker run --rm "$${DOCKER_IMAGE:-taxi-batch:module06}" --help

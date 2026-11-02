# Module 6 — Engineering Best Practices

**Project:** DataTalksClub MLOps Zoomcamp 2025 — NYC Yellow Taxi Duration Prediction  
**Scope:** Unit testing, regression testing, S3 integration, code quality, pre-commit, GitHub Actions, Docker packaging, and Terraform with a **local** S3 emulator.

This is the **single canonical README for Module 6**. The Module 4 batch scorer supplies the trusted pretrained model and the existing Pipenv lockfile; Module 6 does not retrain it.

## Architecture

```text
                         GitHub Actions
                   lint / tests / integration
                              |
                    Terraform (local only)
                              |
   four-row Parquet fixture -> LocalStack S3 bucket
                              |
                     batch.py (Python)
                  prepare -> encode -> predict
                              |
                output Parquet to LocalStack S3
                              |
                     integration assertions
```

At the unit layer, pure feature transformations are tested with `pytest` without S3. At the integration layer, `integration_test.py` uploads a four-row fixture, executes the real scoring CLI, reads the generated S3 Parquet output, and validates prediction values. Terraform provisions a disposable LocalStack bucket for the CI integration job. The Docker image packages the same batch scorer with the verified course model and locked runtime dependencies.

## Repository structure

```text
module06/
├── README.md                         # This is the only README for Module 6
├── Makefile                          # Local development commands
├── pyproject.toml                    # Ruff configuration
├── .pre-commit-config.yaml           # Git hooks
├── Dockerfile.batch                  # Reproducible batch image
├── homework/
│   ├── batch.py                      # Refactored batch scorer
│   ├── integration_test.py           # End-to-end S3 test
│   ├── compose.localstack.yaml       # LocalStack S3 emulator
│   ├── model.sha256                  # Generated locally; commit checksum, not pickle
│   └── tests/                        # Unit tests
├── scripts/
│   ├── build_image.sh                # Minimal Docker build context
│   └── verify_local.sh               # Final local verification
└── terraform/
    ├── versions.tf
    ├── main.tf
    ├── variables.tf
    └── .gitignore                     # Excludes local state and provider cache

.github/workflows/module06-ci.yml    # CI and optional GHCR release
```

The GitHub Actions workflow is at the **repository root**, where GitHub detects it; it is not nested in `module06/`.

## Prerequisites

- Python 3.12; working `pipenv`; Docker Engine and Compose; `make`.
- Terraform CLI >= 1.6 installed on the **host** (not Python). Check with `terraform version`.
- Module 4 homework environment: `module04/homework/Pipfile` and `Pipfile.lock`, already populated with `scikit-learn==1.5.0`, `pytest`, `boto3`, `ruff`, and `pre-commit`.
- Trusted upstream `module04/homework/model.bin`. **Do not commit the pickle**, because unpickling an untrusted file can run arbitrary code.
- A loopback LocalStack S3 endpoint on port `4566` (started below).

All shell commands below are run from the repository root, e.g. `~/work/mlops`.

Set the Pipenv project explicitly to avoid accidentally using the root `.venv`:

```bash
cd ~/work/mlops
export PIPENV_IGNORE_VIRTUALENVS=1
export PIPENV_PIPFILE="$PWD/module04/homework/Pipfile"
pipenv sync --dev
```

If development dependencies were not previously installed, run `pipenv install --dev pytest boto3 ruff pre-commit` from `module04/homework` with the corresponding `PIPENV_PIPFILE`; this updates the homework's Pipfile/lockfile.

### Model checksum

Create the checksum **once** from the course model you already ran successfully:

```bash
sha256sum module04/homework/model.bin > module06/homework/model.sha256
sha256sum --check module06/homework/model.sha256
```

The checksum file stores the model's relative path and SHA-256 digest; it is safe to commit. The GitHub workflow downloads the model from the [official course source](https://raw.githubusercontent.com/DataTalksClub/mlops-zoomcamp/main/cohorts/2025/04-deployment/homework/model.bin) and rejects unexpected bytes. If the upstream file changes, investigate before updating the expected digest.

## Quality and regression testing

```bash
make -f module06/Makefile format       # Applies Ruff fixes intentionally
make -f module06/Makefile check        # Lint + format check + unit tests
make -f module06/Makefile hooks-install
make -f module06/Makefile hooks-check
```

**Observed local results:** eight pytest unit tests passed, Ruff reported no linting problems, and all seven Module 6 Python files were formatted. The refactored March 2023 batch scorer preserved the earlier result:

| March 2023 regression metric | Verified result |
| ---------------------------- | --------------: |
| Predictions                  |       3,316,216 |
| Mean minutes                 |         14.2039 |
| Std. dev. minutes            |          6.2475 |

To repeat the large March run, ensure `module04/homework/data/yellow_tripdata_2023-03.parquet` exists, then run:

```bash
pipenv run python module06/homework/batch.py 2023 3
```

The tests use tiny in-memory fixtures by default; the large regression job is intentionally separate.

## LocalStack S3 integration

The legacy pinned LocalStack image is used only for the coursework. Newer authenticated LocalStack versions require an auth token; do not commit tokens. This Compose configuration uses Linux `network_mode: host` because Docker bridge port-forwarding failed on the development workstation when the VPN was enabled. It binds the gateway to `127.0.0.1:4566`.

```bash
docker compose -f module06/homework/compose.localstack.yaml up -d
curl --noproxy '*' -fsS http://127.0.0.1:4566/_localstack/health
S3_ENDPOINT_URL=http://127.0.0.1:4566 make -f module06/Makefile integration
```

The S3 endpoint is **required to be loopback** in the integration script, so the test cannot accidentally send records to a real AWS endpoint.

**Observed integration result:**

| Check                            |        Actual value |
| -------------------------------- | ------------------: |
| Input fixture uploaded           |         3,273 bytes |
| Valid rows after duration filter |                   2 |
| Mean predicted duration          |     18.1386 minutes |
| Std. dev.                        |      5.0585 minutes |
| Prediction sum                   | **36.2773 minutes** |
| End-to-end verification          |          **Passed** |

The small fixture contains four rows; only two have durations within 1–60 minutes. The uploaded-byte count varies with PyArrow versions. The scorer supports `INPUT_FILE_PATTERN`, `OUTPUT_FILE_PATTERN`, and `S3_ENDPOINT_URL` for S3-compatible paths.

## Terraform — Infrastructure as Code

Terraform provisions one disposable **LocalStack-only** S3 bucket. The AWS provider is configured with fake credentials, region `us-east-1`, an explicit loopback S3 endpoint and credential/metadata validation disabled. **Never use this configuration against a real AWS account.**

```bash
make -f module06/Makefile tf-validate
make -f module06/Makefile tf-plan
make -f module06/Makefile tf-apply
terraform -chdir=module06/terraform output bucket_name
```

Expected bucket: `nyc-duration-iac`. This is intentionally distinct from the original integration test's default `nyc-duration` bucket.

Test the actual Terraform-created bucket:

```bash
S3_BUCKET=nyc-duration-iac \
S3_ENDPOINT_URL=http://127.0.0.1:4566 \
make -f module06/Makefile integration
```

`terraform init` produces `module06/terraform/.terraform.lock.hcl`. **Commit the provider lockfile** for reproducible future installations; never commit `.terraform/`, `terraform.tfstate*`, secret values, or plan files. Terraform state is retained only on the developer's machine in this example.

When finished, optionally destroy the disposable bucket **before** shutting down LocalStack:

```bash
make -f module06/Makefile tf-destroy
docker compose -f module06/homework/compose.localstack.yaml down
```

**Caveat:** The Terraform example provisions an S3 bucket for integration testing, not real AWS Kinesis/Lambda/ECR infrastructure. The latter requires AWS credentials, deployment decisions, and production security design beyond a free local course implementation.

## Container packaging

The build script creates a minimal temporary Docker context containing only `batch.py`, the Pipenv lock, and the checksummed course model. It avoids sending the repository's large datasets to the Docker daemon.

```bash
bash module06/scripts/build_image.sh
docker run --rm taxi-batch:module06 --help
```

If Docker build networking requires the earlier host-network workaround:

```bash
DOCKER_BUILD_NETWORK=host bash module06/scripts/build_image.sh
```

The container uses Python 3.12, a non-root user, and the locked runtime dependencies. The CLI accepts a year and a month; point input and output paths to mounted directories or S3 when running actual batch predictions.

## CI/CD — GitHub Actions

Workflow: `.github/workflows/module06-ci.yml`.

- **Quality:** install locked development dependencies, run Ruff and eight unit tests.
- **Integration:** download and checksum-verify the model; start LocalStack; validate and apply Terraform; execute S3-to-S3 prediction against its provisioned bucket.
- **Package:** build the batch image and smoke-test its CLI.
- **Publish (opt-in):** only via a manual GitHub Actions workflow dispatch with `publish_image=true`, publish the checksummed image to GitHub Container Registry (GHCR), using `GITHUB_TOKEN` and package write permission. Normal pushes and pull requests never publish images.

No production deployment or AWS resources are created by this workflow. A GHCR release is an optional software-delivery artifact, not an automatically deployed live service.

GitHub-hosted execution cannot be verified completely until the workflow is pushed. Before the final single commit, use the local end-to-end command below to verify the same important code paths.

## One-command local verification

With LocalStack started, Terraform installed, checksum created, and Pipenv configured:

```bash
bash module06/scripts/verify_local.sh
```

This runs lint, format check, unit tests, Terraform validation, Terraform apply, LocalStack integration, Docker build and CLI smoke test. It intentionally leaves the local S3 emulator and Terraform state available for examination. It does **not** commit or push anything.

2025 assignment: https://github.com/DataTalksClub/mlops-zoomcamp/blob/main/cohorts/2025/06-best-practices/homework.md


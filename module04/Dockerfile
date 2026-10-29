FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# The model's MLflow-generated requirements preserve its training-time ML stack.
COPY artifacts/model/requirements.txt /tmp/model-requirements.txt
COPY requirements-api.txt /tmp/requirements-api.txt
RUN python -m pip install --no-cache-dir \
    -r /tmp/model-requirements.txt \
    -r /tmp/requirements-api.txt

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app app

# Model/vectorizer are bundled in the image; no runtime MLflow server required.
COPY --chown=app:app api.py /app/module04/api.py
COPY --chown=app:app artifacts/ /app/module04/artifacts/

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
  CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health', timeout=3).read()" || exit 1

CMD ["python", "-m", "uvicorn", "module04.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

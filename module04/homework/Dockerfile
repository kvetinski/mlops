
FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

RUN python -m pip install --no-cache-dir \
    numpy==1.26.4 \
    scikit-learn==1.5.0 \
    pandas==2.2.3 \
    pyarrow==17.0.0

COPY score.py /app/score.py

ENTRYPOINT ["python", "/app/score.py"]

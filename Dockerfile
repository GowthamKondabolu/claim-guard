FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CLAIMGUARD_MODEL_PATH=/app/artifacts/isolation_forest.joblib

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY artifacts ./artifacts

EXPOSE 8000
CMD ["uvicorn", "claimguard.api.main:app", "--host", "0.0.0.0", "--port", "8000"]


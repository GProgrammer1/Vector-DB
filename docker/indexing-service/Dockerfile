FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -e ".[dev]" || \
    pip install --no-cache-dir \
    numpy>=1.24.0 \
    scipy>=1.10.0 \
    sentence-transformers>=2.2.0 \
    transformers>=4.30.0 \
    tokenizers>=0.13.0 \
    fastapi>=0.100.0 \
    uvicorn[standard]>=0.20.0 \
    pydantic>=2.0.0 \
    PyYAML>=6.0.0

COPY src/config.yaml ./config.yaml

EXPOSE 8000

CMD ["uvicorn", "vector_db.api.app:app", "--host", "0.0.0.0", "--port", "8000"]


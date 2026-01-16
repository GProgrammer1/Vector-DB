FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install base dependencies (indexing service doesn't need PyTorch)
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e .

COPY src/config.yaml ./config.yaml

EXPOSE 8000

CMD ["uvicorn", "vector_db.api.app:app", "--host", "0.0.0.0", "--port", "8000"]


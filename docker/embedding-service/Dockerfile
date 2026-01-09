FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir -e ".[dev]" || \
    pip install --no-cache-dir \
    numpy>=1.24.0 \
    scipy>=1.10.0 \
    torch>=2.0.0 \
    sentence-transformers>=2.2.0 \
    transformers>=4.30.0 \
    tokenizers>=0.13.0 \
    fastapi>=0.100.0 \
    uvicorn[standard]>=0.20.0 \
    pydantic>=2.0.0 \
    PyYAML>=6.0.0

COPY docker/embedding-service/app.py ./embedding_app.py


EXPOSE 8001

CMD ["uvicorn", "embedding_app:app", "--host", "0.0.0.0", "--port", "8001"]


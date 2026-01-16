FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install base dependencies first (without PyTorch)
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e .

# Install PyTorch CPU version (default for Docker)
# For GPU support, override this with build args or use a different base image
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir torch torchvision --index-url ${PYTORCH_INDEX_URL}

# Install CPU extras (sentence-transformers, transformers, tokenizers)
RUN pip install --no-cache-dir -e ".[cpu]"

COPY docker/embedding-service/app.py ./embedding_app.py
COPY src/config.yaml ./config.yaml

# Pre-download the embedding model to avoid startup delays and network issues
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

EXPOSE 8001

CMD ["uvicorn", "embedding_app:app", "--host", "0.0.0.0", "--port", "8001"]


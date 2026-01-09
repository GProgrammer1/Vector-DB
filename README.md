# Vector Database Engine

A high-performance vector database engine built from scratch with support for approximate nearest neighbor search, product quantization, and memory-mapped storage.

## Architecture

### Decoupled Service Architecture

The system uses a **decoupled microservices architecture** that separates GPU-intensive embedding generation from CPU-bound indexing operations, enabling independent scaling:

- **Embedding Service** (Port 8001): GPU-optimized service for generating embeddings using sentence-transformers
  - Can be scaled independently based on GPU availability
  - Handles batch embedding requests
  - Stateless and horizontally scalable

- **Indexing Service** (Port 8000): CPU-bound service for indexing and search operations
  - Manages HNSW index lifecycle
  - Handles storage and retrieval
  - Communicates with embedding service via HTTP API
  - Can be scaled independently based on CPU/memory needs

### Key Features

- **HNSW Indexing**: Hierarchical Navigable Small World graph for fast approximate nearest neighbor search
- **Memory-Mapped Storage**: Two-layer memmap storage for efficient disk-backed persistence
- **Product Quantization**: Compression technique for reducing memory footprint
- **Threshold-Based Persistence**: Automatic index flushing when memory threshold is reached
- **Docker Containerization**: Fully containerized services with docker-compose orchestration

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Start both services
docker-compose up -d

# Scale services independently
docker-compose up -d --scale embedding-service=3 --scale indexing-service=2

# Check service health
curl http://localhost:8001/health  # Embedding service
curl http://localhost:8000/health  # Indexing service
```

### Manual Setup

1. Install dependencies:
```bash
pip install -e ".[dev]"
```

2. Start embedding service:
```bash
cd docker/embedding-service
uvicorn app:app --host 0.0.0.0 --port 8001
```

3. Start indexing service:
```bash
export USE_EMBEDDING_SERVICE=true
export EMBEDDING_SERVICE_URL=http://localhost:8001
uvicorn src.vector_db.api.app:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### Indexing Service (Port 8000)

- `POST /embed` - Embed a document and add to index
- `GET /health` - Health check with index status

### Embedding Service (Port 8001)

- `POST /embed` - Embed a single text
- `POST /embed/batch` - Embed multiple texts
- `GET /health` - Health check

## Configuration

Edit `src/config.yaml` to configure:

- Embedding model and dimension
- Index parameters (M, ef_construction, flush_threshold)
- Storage capacity and file paths

## Development

```bash
# Run tests
pytest tests/

# Type checking
mypy src/

# Linting
ruff check src/
```

## Scaling Strategy

### GPU Scaling (Embedding Service)
- Scale embedding service instances based on GPU availability
- Each instance can handle multiple concurrent requests
- Use load balancer for distribution

### CPU Scaling (Indexing Service)
- Scale indexing service instances based on query load
- Each instance maintains its own index copy or uses shared storage
- Consider read replicas for high query throughput


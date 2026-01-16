# Vector Database Engine

A high-performance vector database engine built from scratch based on memory mapped storage, supporting 
both HNSW indexing and IVF indexing with product quantization and integrated in a comprehensive FastAPI service that supports data storage and retrieval.

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

## Installation

### Prerequisites

- Python 3.9 or higher
- Virtual environment (recommended)
- For GPU support: compatible GPU drivers installed

### Step 1: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows
```

### Step 2: Install Base Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -e .
```

This installs all core dependencies (numpy, scipy, fastapi, etc.) without PyTorch.

### Step 3: Install PyTorch Based on Your Device

PyTorch must be installed separately based on your hardware. Choose the appropriate option below.

#### CPU-Only Installation

For systems without GPU or for CPU-only training:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Then install the CPU extras:

```bash
pip install -e ".[cpu]"
```

#### CUDA Installation (NVIDIA GPU)

**Version Compatibility Matrix:**

| PyTorch Version | CUDA 11.8 | CUDA 12.1 | CUDA 12.4 | Python Support |
|----------------|-----------|-----------|-----------|----------------|
| 2.0.x          | Supported | Not Supported | Not Supported | 3.8-3.11      |
| 2.1.x          | Supported | Supported | Not Supported | 3.8-3.11      |
| 2.2.x          | Supported | Supported | Supported | 3.8-3.12      |
| 2.3.x          | Supported | Supported | Supported | 3.9-3.12      |
| 2.4.x+         | Supported | Supported | Supported | 3.9-3.12      |

**Installation Commands:**

For CUDA 11.8:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

For CUDA 12.1:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

For CUDA 12.4:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Then install the GPU extras:

```bash
pip install -e ".[gpu]"
```

#### ROCm Installation (AMD GPU)

**Version Compatibility Matrix:**

| PyTorch Version | ROCm 5.7 | ROCm 6.0 | ROCm 6.1 | Python Support |
|----------------|----------|----------|----------|----------------|
| 2.0.x          | Supported | Not Supported | Not Supported | 3.8-3.11      |
| 2.1.x          | Supported | Supported | Not Supported | 3.8-3.11      |
| 2.2.x          | Supported | Supported | Supported | 3.8-3.12      |
| 2.3.x          | Supported | Supported | Supported | 3.9-3.12      |
| 2.4.x+         | Supported | Supported | Supported | 3.9-3.12      |

**Installation Commands:**

For ROCm 5.7:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.7
```

For ROCm 6.0:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
```

For ROCm 6.1:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.1
```

Then install the GPU extras:

```bash
pip install -e ".[gpu]"
```

#### Apple Silicon (MPS) Installation

**Version Compatibility Matrix:**

| PyTorch Version | macOS Version | Apple Silicon | Python Support |
|----------------|---------------|---------------|----------------|
| 2.0.x          | 12.3+         | M1, M1 Pro/Max | 3.8-3.11      |
| 2.1.x          | 12.3+         | M1, M1 Pro/Max, M2 | 3.8-3.11      |
| 2.2.x          | 12.3+         | M1, M2, M3    | 3.8-3.12      |
| 2.3.x          | 12.3+         | M1, M2, M3, M4 | 3.9-3.12      |
| 2.4.x+         | 12.3+         | M1, M2, M3, M4 | 3.9-3.12      |

**Installation Command:**

```bash
pip install torch torchvision
```

PyTorch automatically detects Apple Silicon and uses Metal Performance Shaders (MPS) backend.

Then install the GPU extras:

```bash
pip install -e ".[gpu]"
```

### Step 4: Install Development Dependencies (Optional)

For development, testing, and code formatting:

```bash
pip install -e ".[dev]"
```

### Complete Installation Examples

**CPU-only:**
```bash
pip install -e .
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[cpu]"
```

**CUDA 12.1:**
```bash
pip install -e .
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[gpu]"
```

**Apple Silicon:**
```bash
pip install -e .
pip install torch torchvision
pip install -e ".[gpu]"
```

### Troubleshooting

**Issue: `torch.cuda.is_available()` returns `False`**
- Verify CUDA drivers: `nvidia-smi`
- Ensure PyTorch CUDA version matches installed CUDA toolkit
- Reinstall PyTorch with correct CUDA version

**Issue: Import errors after installation**
- Ensure virtual environment is activated
- Try: `pip install --upgrade pip setuptools wheel`
- Reinstall: `pip uninstall torch torchvision && pip install ...`

**Issue: ROCm not detected**
- Verify ROCm installation: `rocminfo`
- Check environment variables: `HIP_PATH`, `ROCM_PATH`
- Ensure compatible AMD GPU and drivers

**Issue: MPS not available on macOS**
- Update macOS to 12.3 or later
- Ensure you're using Apple Silicon (M1/M2/M3/M4)
- Update PyTorch to latest version

## Quick Start

### Using Docker Compose (Recommended)

#### CPU Version (Default)

```bash
# Start both services (CPU version by default)
docker-compose up -d

# Scale services independently
docker-compose up -d --scale embedding-service=3 --scale indexing-service=2

# Check service health
curl http://localhost:8001/health  # Embedding service
curl http://localhost:8000/health  # Indexing service
```

#### GPU Version (CUDA)

To build with GPU support, use the `PYTORCH_INDEX_URL` build argument:

```bash
# Build and start with CUDA 12.1 support
docker-compose build --build-arg PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 embedding-service
docker-compose up -d
```

For other CUDA versions, replace `cu121` with:
- `cu118` for CUDA 11.8
- `cu124` for CUDA 12.4

**Note:** Ensure your Docker environment has GPU access configured (e.g., `nvidia-docker` or Docker Desktop with GPU support).

> [!NOTE]
> The Dockerfiles install the CPU version of PyTorch by default. For GPU support, build with the appropriate `PYTORCH_INDEX_URL` build argument.

### Using Podman

If you are using Podman, you can use `podman-compose`:

```bash
# Start both services
podman-compose up -d --build

# Check status
podman ps
```

### Manual Setup

#### Step 1: Install Dependencies

Follow the [Installation](#installation) steps above. For a quick start with CPU:

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Install base dependencies
pip install --upgrade pip setuptools wheel
pip install -e .

# Install PyTorch (CPU version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install CPU extras (sentence-transformers, transformers, tokenizers)
pip install -e ".[cpu]"
```

For GPU support, see the [Installation](#installation) section for CUDA/ROCm/MPS instructions.

#### Step 2: Start the Embedding Service

In the first terminal:

```bash
# Set the config path
export CONFIG_PATH=src/config.yaml

# Start the embedding service
cd docker/embedding-service
uvicorn app:app --host 0.0.0.0 --port 8001
```

#### Step 3: Start the Indexing Service

In a second terminal:

```bash
# Activate the same virtual environment
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Set required environment variables
export EMBEDDING_SERVICE_URL=http://localhost:8001
export CONFIG_PATH=src/config.yaml

# Start the indexing service
uvicorn src.vector_db.api.app:app --host 0.0.0.0 --port 8000
```

#### Step 4: Verify and Test

Check that both services are running:

```bash
# Check embedding service
curl http://localhost:8001/health

# Check indexing service
curl http://localhost:8000/health
```

Test the API:

```bash
# Embed a document
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"content": "This is a test document", "metadata": {"source": "test"}}'

# Search for similar documents
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "top_k": 5}'
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


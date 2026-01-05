# Vector DB

A high-performance Vector Database engine built from scratch with support for embeddings, similarity search, and product quantization.

## Features

- **Vector Storage & Indexing**: Efficient storage and retrieval of high-dimensional vectors
- **Similarity Search**: Fast approximate nearest neighbor (ANN) search
- **Product Quantization**: Advanced compression techniques for large-scale vector databases
- **REST API**: FastAPI-based API for easy integration
- **Embedding Support**: Integration with transformers for generating embeddings

## Installation

```bash
# Install the package in editable mode (CPU version by default)
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"

# Install with all optional dependencies
pip install -e ".[all]"
```

### GPU Support

The package works with both CPU and GPU. By default, it installs the CPU version of PyTorch (no NVIDIA dependencies required).

**For NVIDIA GPU users (optional):**
```bash
# First install PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Then install the package
pip install -e .
```

The code automatically detects and uses the best available device (CPU, CUDA, or Apple Silicon MPS). You can also explicitly specify the device:

```python
from vector_db.inference import EmbeddingService, get_device, get_device_info
from sentence_transformers import SentenceTransformer

# Check available devices
print(get_device_info())

# Auto-detect best device (default)
model = SentenceTransformer("all-MiniLM-L6-v2")
service = EmbeddingService(model, device="auto")

# Force CPU
service = EmbeddingService(model, device="cpu")

# Use GPU if available
service = EmbeddingService(model, device="cuda")
```

## Development

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=vector_db

# Format code
ruff format .

# Lint code
ruff check .

# Type checking
mypy src/
```

## License

MIT


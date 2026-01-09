"""Unit tests for EmbeddingService."""

import numpy as np
import pytest
import tempfile
from pathlib import Path

try:
    from vector_db.services.embedding_service import EmbeddingService
    _EMBEDDING_AVAILABLE = True
except ImportError:
    _EMBEDDING_AVAILABLE = False
    EmbeddingService = None  # type: ignore[assignment, misc]

pytestmark = pytest.mark.skipif(
    not _EMBEDDING_AVAILABLE,
    reason="sentence-transformers not installed"
)


@pytest.fixture
def dummy_config_file():
    """Create a dummy config file for tests."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".yaml") as f:
        f.write("""
embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  dimension: 384
device: CPU
index:
  ef_construction: 200
  M: 16
vector_db:
  file_path: ../vector_db
  dimension: 384
  capacity: 1000000
        """)
        config_path = f.name
    yield config_path
    Path(config_path).unlink(missing_ok=True)


class TestEmbeddingService:
    """Test suite for EmbeddingService."""

    def test_init(self, dummy_config_file):
        """Test service initialization."""
        service = EmbeddingService(config_path=dummy_config_file)
        assert service.dim == 384

    def test_embed_text(self, dummy_config_file):
        """Test embedding a single text."""
        service = EmbeddingService(config_path=dummy_config_file)
        embedding = service.embed_text("Hello, world!")
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        assert embedding.dtype == np.float32

    def test_embed_texts(self, dummy_config_file):
        """Test embedding multiple texts."""
        service = EmbeddingService(config_path=dummy_config_file)
        texts = ["Hello", "World", "Test"]
        embeddings = service.embed_texts(texts)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 384)
        assert embeddings.dtype == np.float32

    def test_dimension_property(self, dummy_config_file):
        """Test dimension property."""
        service = EmbeddingService(config_path=dummy_config_file)
        assert service.dim == 384


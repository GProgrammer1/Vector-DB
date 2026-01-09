"""Unit tests for EmbeddingService."""

import numpy as np
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

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

    @patch('vector_db.services.embedding_service._SENTENCE_TRANSFORMERS_AVAILABLE', True)
    @patch('vector_db.services.embedding_service.SentenceTransformer')
    @patch('vector_db.services.embedding_service.BaseEmbeddingService')
    def test_init(self, mock_base_service, mock_sentence_transformer, dummy_config_file):
        """Test service initialization."""
        # Mock SentenceTransformer
        mock_model = Mock()
        mock_model.to = Mock(return_value=mock_model)
        mock_sentence_transformer.return_value = mock_model
        
        # Mock BaseEmbeddingService
        mock_base_instance = Mock()
        mock_base_service.return_value = mock_base_instance
        
        service = EmbeddingService(config_path=dummy_config_file)
        assert service.dim == 384

    @patch('vector_db.services.embedding_service._SENTENCE_TRANSFORMERS_AVAILABLE', True)
    @patch('vector_db.services.embedding_service.SentenceTransformer')
    @patch('vector_db.services.embedding_service.BaseEmbeddingService')
    def test_embed_text(self, mock_base_service, mock_sentence_transformer, dummy_config_file):
        """Test embedding a single text."""
        # Mock SentenceTransformer
        mock_model = Mock()
        mock_model.to = Mock(return_value=mock_model)
        mock_sentence_transformer.return_value = mock_model
        
        # Mock the base embedding service
        mock_service_instance = Mock()
        expected_embedding = np.random.rand(384).astype(np.float32)
        mock_service_instance.embed_text = Mock(return_value=expected_embedding)
        mock_base_service.return_value = mock_service_instance
        
        service = EmbeddingService(config_path=dummy_config_file)
        embedding = service.embed_text("Hello, world!")
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)
        assert embedding.dtype == np.float32
        mock_service_instance.embed_text.assert_called_once_with("Hello, world!")

    @patch('vector_db.services.embedding_service._SENTENCE_TRANSFORMERS_AVAILABLE', True)
    @patch('vector_db.services.embedding_service.SentenceTransformer')
    @patch('vector_db.services.embedding_service.BaseEmbeddingService')
    def test_embed_texts(self, mock_base_service, mock_sentence_transformer, dummy_config_file):
        """Test embedding multiple texts."""
        # Mock SentenceTransformer
        mock_model = Mock()
        mock_model.to = Mock(return_value=mock_model)
        mock_sentence_transformer.return_value = mock_model
        
        # Mock the base embedding service
        mock_service_instance = Mock()
        expected_embeddings = np.random.rand(3, 384).astype(np.float32)
        mock_service_instance.embed_texts = Mock(return_value=expected_embeddings)
        mock_base_service.return_value = mock_service_instance
        
        service = EmbeddingService(config_path=dummy_config_file)
        texts = ["Hello", "World", "Test"]
        embeddings = service.embed_texts(texts)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (3, 384)
        assert embeddings.dtype == np.float32
        mock_service_instance.embed_texts.assert_called_once_with(texts)

    @patch('vector_db.services.embedding_service._SENTENCE_TRANSFORMERS_AVAILABLE', True)
    @patch('vector_db.services.embedding_service.SentenceTransformer')
    @patch('vector_db.services.embedding_service.BaseEmbeddingService')
    def test_dimension_property(self, mock_base_service, mock_sentence_transformer, dummy_config_file):
        """Test dimension property."""
        # Mock SentenceTransformer
        mock_model = Mock()
        mock_model.to = Mock(return_value=mock_model)
        mock_sentence_transformer.return_value = mock_model
        
        # Mock BaseEmbeddingService
        mock_base_instance = Mock()
        mock_base_service.return_value = mock_base_instance
        
        service = EmbeddingService(config_path=dummy_config_file)
        assert service.dim == 384


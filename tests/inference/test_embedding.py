"""Tests for EmbeddingService."""

import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock

# Mock sentence_transformers before importing
import sys
mock_sentence_transformers = MagicMock()
sys.modules['sentence_transformers'] = mock_sentence_transformers

from vector_db.inference.embedding import EmbeddingService


class TestEmbeddingService:
    """Test suite for EmbeddingService."""

    def test_init_with_valid_model(self):
        """Test initialization with a valid model."""
        mock_model = Mock()
        mock_model.to = Mock(return_value=mock_model)
        
        with patch("vector_db.inference.embedding.get_device", return_value="cpu"):
            service = EmbeddingService(mock_model, device="cpu")
            
            assert service.model == mock_model
            assert service.device == "cpu"
            mock_model.to.assert_called_once_with("cpu")

    def test_init_with_none_model(self):
        """Test initialization with None model raises ValueError."""
        with pytest.raises(ValueError, match="Model is required"):
            EmbeddingService(None)

    def test_init_with_auto_device(self):
        """Test initialization with auto device selection."""
        mock_model = Mock()
        mock_model.to = Mock(return_value=mock_model)
        
        with patch("vector_db.inference.embedding.get_device", return_value="cpu") as mock_get_device:
            service = EmbeddingService(mock_model, device="auto")
            
            mock_get_device.assert_called_once_with("auto")
            assert service.device == "cpu"
            mock_model.to.assert_called_once_with("cpu")

    def test_init_with_cpu_device(self):
        """Test initialization with explicit CPU device."""
        mock_model = Mock()
        mock_model.to = Mock(return_value=mock_model)
        
        with patch("vector_db.inference.embedding.get_device", return_value="cpu"):
            service = EmbeddingService(mock_model, device="cpu")
            
            assert service.device == "cpu"
            mock_model.to.assert_called_once_with("cpu")

    def test_init_with_cuda_device(self):
        """Test initialization with CUDA device."""
        mock_model = Mock()
        mock_model.to = Mock(return_value=mock_model)
        
        with patch("vector_db.inference.embedding.get_device", return_value="cuda"):
            service = EmbeddingService(mock_model, device="cuda")
            
            assert service.device == "cuda"
            mock_model.to.assert_called_once_with("cuda")

    def test_embed_text(self):
        """Test embedding a single text."""
        mock_model = Mock()
        mock_model.to = Mock(return_value=mock_model)
        expected_embedding = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        mock_model.encode = Mock(return_value=expected_embedding)
        
        with patch("vector_db.inference.embedding.get_device", return_value="cpu"):
            service = EmbeddingService(mock_model, device="cpu")
            result = service.embed_text("test text")
            
            assert isinstance(result, np.ndarray)
            np.testing.assert_array_equal(result, expected_embedding)
            mock_model.encode.assert_called_once_with("test text", device="cpu")

    def test_embed_texts(self):
        """Test embedding multiple texts."""
        mock_model = Mock()
        mock_model.to = Mock(return_value=mock_model)
        expected_embeddings = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ])
        mock_model.encode = Mock(return_value=expected_embeddings)
        
        with patch("vector_db.inference.embedding.get_device", return_value="cpu"):
            service = EmbeddingService(mock_model, device="cpu")
            texts = ["text1", "text2"]
            result = service.embed_texts(texts)
            
            assert isinstance(result, np.ndarray)
            assert result.shape == (2, 3)
            np.testing.assert_array_equal(result, expected_embeddings)
            mock_model.encode.assert_called_once_with(texts, device="cpu")

    def test_embed_texts_empty_list(self):
        """Test embedding empty list of texts."""
        mock_model = Mock()
        mock_model.to = Mock(return_value=mock_model)
        expected_embeddings = np.array([]).reshape(0, 384)  # Empty array
        mock_model.encode = Mock(return_value=expected_embeddings)
        
        with patch("vector_db.inference.embedding.get_device", return_value="cpu"):
            service = EmbeddingService(mock_model, device="cpu")
            result = service.embed_texts([])
            
            assert isinstance(result, np.ndarray)
            mock_model.encode.assert_called_once_with([], device="cpu")

    def test_embed_text_with_different_devices(self):
        """Test that device parameter is passed correctly to encode."""
        mock_model = Mock()
        mock_model.to = Mock(return_value=mock_model)
        mock_model.encode = Mock(return_value=np.array([0.1, 0.2, 0.3]))
        
        # Test with CPU
        with patch("vector_db.inference.embedding.get_device", return_value="cpu"):
            service = EmbeddingService(mock_model, device="cpu")
            service.embed_text("test")
            mock_model.encode.assert_called_with("test", device="cpu")
        
        # Test with CUDA
        mock_model.reset_mock()
        mock_model.to = Mock(return_value=mock_model)
        with patch("vector_db.inference.embedding.get_device", return_value="cuda"):
            service = EmbeddingService(mock_model, device="cuda")
            service.embed_text("test")
            mock_model.encode.assert_called_with("test", device="cuda")


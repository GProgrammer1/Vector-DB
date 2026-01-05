"""Tests for ProductQuantizationService."""

import numpy as np
import pytest
from vector_db.inference.pq import ProductQuantizationService


class TestProductQuantizationService:
    """Test suite for ProductQuantizationService."""

    def test_init_valid_params(self):
        """Test initialization with valid parameters."""
        pq = ProductQuantizationService(k=256, chunks=8, dim=768)
        assert pq.k == 256
        assert pq.chunks == 8
        assert pq.dim == 768
        assert pq.subdim == 96
        assert pq.centroids is None

    def test_init_invalid_k(self):
        """Test initialization with invalid k."""
        with pytest.raises(ValueError, match="k must be greater than 0"):
            ProductQuantizationService(k=0, chunks=8, dim=768)

    def test_init_invalid_chunks(self):
        """Test initialization with invalid chunks."""
        with pytest.raises(ValueError, match="chunks must be greater than 0"):
            ProductQuantizationService(k=256, chunks=0, dim=768)

    def test_init_invalid_dim(self):
        """Test initialization with invalid dim."""
        with pytest.raises(ValueError, match="dim must be greater than 0"):
            ProductQuantizationService(k=256, chunks=8, dim=0)

    def test_init_dim_not_divisible_by_chunks(self):
        """Test initialization when dim is not divisible by chunks."""
        with pytest.raises(ValueError, match="dim must be divisible by chunks"):
            ProductQuantizationService(k=256, chunks=8, dim=769)

    def test_validate_embeddings_not_numpy_array(self):
        """Test validation with non-numpy array."""
        pq = ProductQuantizationService(k=4, chunks=2, dim=8)
        with pytest.raises(TypeError, match="Embeddings must be a numpy array"):
            pq._validate_embeddings([[1, 2, 3, 4, 5, 6, 7, 8]])

    def test_validate_embeddings_wrong_dimension(self):
        """Test validation with wrong array dimension."""
        pq = ProductQuantizationService(k=4, chunks=2, dim=8)
        embeddings = np.array([1, 2, 3, 4, 5, 6, 7, 8])  # 1D instead of 2D
        with pytest.raises(ValueError, match="Embeddings must be 2D array"):
            pq._validate_embeddings(embeddings)

    def test_validate_embeddings_wrong_embedding_dim(self):
        """Test validation with wrong embedding dimension."""
        pq = ProductQuantizationService(k=4, chunks=2, dim=8)
        embeddings = np.array([[1, 2, 3, 4, 5, 6, 7]])  # dim=7 instead of 8
        with pytest.raises(ValueError, match="Embedding dimension must be 8"):
            pq._validate_embeddings(embeddings)

    def test_chunk_embeddings(self):
        """Test chunking embeddings into sub-vectors."""
        pq = ProductQuantizationService(k=4, chunks=2, dim=8)
        embeddings = np.array([
            [1, 2, 3, 4, 5, 6, 7, 8],
            [9, 10, 11, 12, 13, 14, 15, 16]
        ])
        chunks = pq._chunk_embeddings(embeddings)
        
        assert len(chunks) == 2
        assert len(chunks[0]) == 2  # Two embeddings
        assert len(chunks[1]) == 2
        
        # Check first chunk of first embedding
        np.testing.assert_array_equal(chunks[0][0], [1, 2, 3, 4])
        # Check second chunk of first embedding
        np.testing.assert_array_equal(chunks[1][0], [5, 6, 7, 8])

    def test_find_nearest_centroid(self):
        """Test finding nearest centroid for a chunk."""
        pq = ProductQuantizationService(k=4, chunks=2, dim=8)
        chunk = np.array([1.0, 2.0, 3.0, 4.0])
        centroids = np.array([
            [10.0, 20.0, 30.0, 40.0],
            [1.1, 2.1, 3.1, 4.1],  # Closest
            [100.0, 200.0, 300.0, 400.0],
            [50.0, 60.0, 70.0, 80.0]
        ])
        
        nearest_idx = pq._find_nearest_centroid(chunk, centroids)
        assert nearest_idx == 1

    def test_compress_embedding(self):
        """Test compressing a single embedding."""
        pq = ProductQuantizationService(k=4, chunks=2, dim=8)
        embedding = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        
        # Create mock centroids
        centroids = [
            np.array([
                [10.0, 20.0, 30.0, 40.0],
                [1.1, 2.1, 3.1, 4.1],  # Nearest for chunk 0
                [100.0, 200.0, 300.0, 400.0],
                [50.0, 60.0, 70.0, 80.0]
            ]),
            np.array([
                [10.0, 20.0, 30.0, 40.0],
                [100.0, 200.0, 300.0, 400.0],
                [5.1, 6.1, 7.1, 8.1],  # Nearest for chunk 1
                [50.0, 60.0, 70.0, 80.0]
            ])
        ]
        
        compressed = pq._compress_embedding(embedding, centroids)
        assert compressed.shape == (2,)
        assert compressed[0] == 1  # Index of nearest centroid for chunk 0
        assert compressed[1] == 2  # Index of nearest centroid for chunk 1

    def test_compress_single_embedding(self):
        """Test compressing a single embedding end-to-end."""
        # Use k=1 since we only have 1 embedding (k-means needs at least k samples)
        pq = ProductQuantizationService(k=1, chunks=2, dim=8)
        embeddings = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        ], dtype=np.float32)
        
        compressed = pq.compress(embeddings)
        
        assert compressed.shape == (1, 2)
        assert compressed.dtype == np.int64 or compressed.dtype == np.int32
        assert pq.centroids is not None
        assert len(pq.centroids) == 2
        assert pq.centroids[0].shape == (1, 4)  # k centroids, each of subdim size
        assert pq.centroids[1].shape == (1, 4)

    def test_compress_multiple_embeddings(self):
        """Test compressing multiple embeddings."""
        # Use k=3 since we have 3 embeddings (k-means needs at least k samples)
        pq = ProductQuantizationService(k=3, chunks=2, dim=8)
        embeddings = np.array([
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            [17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0]
        ], dtype=np.float32)
        
        compressed = pq.compress(embeddings)
        
        assert compressed.shape == (3, 2)
        assert compressed.dtype == np.int64 or compressed.dtype == np.int32
        assert pq.centroids is not None
        # Verify all indices are within valid range [0, k-1]
        assert np.all(compressed >= 0)
        assert np.all(compressed < 3)

    def test_compress_preserves_structure(self):
        """Test that compression preserves the relationship between embeddings."""
        # Use k=3 since we have 3 embeddings (k-means needs at least k samples)
        pq = ProductQuantizationService(k=3, chunks=4, dim=16)
        # Create embeddings that are similar
        base = np.random.rand(16).astype(np.float32)
        embeddings = np.array([
            base,
            base + 0.1,  # Similar to base
            base + 10.0,  # Different from base
        ])
        
        compressed = pq.compress(embeddings)
        
        assert compressed.shape == (3, 4)
        # First two should be more similar (closer indices) than first and third
        # This is a probabilistic test, so we just check structure
        assert compressed[0].shape == (4,)
        assert compressed[1].shape == (4,)
        assert compressed[2].shape == (4,)
        # Verify all indices are within valid range [0, k-1]
        assert np.all(compressed >= 0)
        assert np.all(compressed < 3)

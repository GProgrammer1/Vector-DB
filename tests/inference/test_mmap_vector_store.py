"""Tests for MemoryMappingService."""

import numpy as np
import pytest
import tempfile
import os
from pathlib import Path

from vector_db.inference.mmap_vector_store import MemoryMappingService


class TestMemoryMappingService:
    """Test suite for MemoryMappingService."""

    def test_init_valid_params(self):
        """Test initialization with valid parameters."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            service = MemoryMappingService(tmp_path, dim=128, capacity=1000)
            assert service.dim == 128
            assert service.capacity == 1000
            assert service.size == 0
            assert service.file_path == Path(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_init_invalid_dim(self):
        """Test initialization with invalid dimension."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            with pytest.raises(ValueError, match="Dimension must be greater than 0"):
                MemoryMappingService(tmp_path, dim=0, capacity=1000)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_init_invalid_capacity(self):
        """Test initialization with invalid capacity."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            with pytest.raises(ValueError, match="Capacity must be greater than 0"):
                MemoryMappingService(tmp_path, dim=128, capacity=0)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_write_single_embedding(self):
        """Test writing a single embedding."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            service = MemoryMappingService(tmp_path, dim=4, capacity=10)
            embedding = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
            
            idx = service.write(embedding)
            
            assert idx == 0
            assert service.size == 1
            # Verify the embedding was written correctly
            np.testing.assert_array_equal(service.file[0], embedding)
        finally:
            os.unlink(tmp_path)

    def test_write_multiple_embeddings(self):
        """Test writing multiple embeddings."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            service = MemoryMappingService(tmp_path, dim=4, capacity=10)
            embeddings = [
                np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32),
                np.array([9.0, 10.0, 11.0, 12.0], dtype=np.float32),
            ]
            
            indices = [service.write(emb) for emb in embeddings]
            
            assert indices == [0, 1, 2]
            assert service.size == 3
            # Verify all embeddings were written correctly
            for i, emb in enumerate(embeddings):
                np.testing.assert_array_equal(service.file[i], emb)
        finally:
            os.unlink(tmp_path)

    def test_write_invalid_type(self):
        """Test writing with invalid type."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            service = MemoryMappingService(tmp_path, dim=4, capacity=10)
            with pytest.raises(TypeError, match="Embedding must be a numpy array"):
                service.write([1.0, 2.0, 3.0, 4.0])
        finally:
            os.unlink(tmp_path)

    def test_write_wrong_dtype(self):
        """Test writing with wrong dtype."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            service = MemoryMappingService(tmp_path, dim=4, capacity=10)
            embedding = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
            with pytest.raises(TypeError, match="Embedding must be a float32 array"):
                service.write(embedding)
        finally:
            os.unlink(tmp_path)

    def test_write_wrong_dimension(self):
        """Test writing with wrong array dimension."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            service = MemoryMappingService(tmp_path, dim=4, capacity=10)
            # 2D array instead of 1D
            embedding = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
            with pytest.raises(ValueError, match="Embedding must be a 1D array"):
                service.write(embedding)
        finally:
            os.unlink(tmp_path)

    def test_write_wrong_size(self):
        """Test writing with wrong embedding size."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            service = MemoryMappingService(tmp_path, dim=4, capacity=10)
            embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)  # size 3 instead of 4
            with pytest.raises(ValueError, match="Embedding must be of dimension 4"):
                service.write(embedding)
        finally:
            os.unlink(tmp_path)

    def test_write_capacity_exceeded(self):
        """Test writing when capacity is exceeded."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            service = MemoryMappingService(tmp_path, dim=4, capacity=2)
            embeddings = [
                np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32),
            ]
            
            # Write up to capacity
            for emb in embeddings:
                service.write(emb)
            
            # Try to write one more
            with pytest.raises(RuntimeError, match="Memory-mapped file is full"):
                service.write(np.array([9.0, 10.0, 11.0, 12.0], dtype=np.float32))
        finally:
            os.unlink(tmp_path)

    def test_read_single_embedding(self):
        """Test reading a single embedding."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            service = MemoryMappingService(tmp_path, dim=4, capacity=10)
            embedding = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
            idx = service.write(embedding)
            
            read_embedding = service.read(idx)
            
            assert isinstance(read_embedding, np.ndarray)
            np.testing.assert_array_equal(read_embedding, embedding)
        finally:
            os.unlink(tmp_path)

    def test_read_multiple_embeddings(self):
        """Test reading multiple embeddings."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            service = MemoryMappingService(tmp_path, dim=4, capacity=10)
            embeddings = [
                np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32),
                np.array([9.0, 10.0, 11.0, 12.0], dtype=np.float32),
            ]
            
            indices = [service.write(emb) for emb in embeddings]
            
            # Read all embeddings
            for i, expected_emb in enumerate(embeddings):
                read_emb = service.read(i)
                np.testing.assert_array_equal(read_emb, expected_emb)
        finally:
            os.unlink(tmp_path)

    def test_read_invalid_index_type(self):
        """Test reading with invalid index type."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            service = MemoryMappingService(tmp_path, dim=4, capacity=10)
            with pytest.raises(TypeError, match="Index must be an integer"):
                service.read("0")
        finally:
            os.unlink(tmp_path)

    def test_read_negative_index(self):
        """Test reading with negative index."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            service = MemoryMappingService(tmp_path, dim=4, capacity=10)
            embedding = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
            service.write(embedding)
            
            with pytest.raises(IndexError):
                service.read(-1)
        finally:
            os.unlink(tmp_path)

    def test_read_index_out_of_bounds(self):
        """Test reading with index out of bounds."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            service = MemoryMappingService(tmp_path, dim=4, capacity=10)
            embedding = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
            service.write(embedding)
            
            with pytest.raises(IndexError, match="Index must be between 0 and 0"):
                service.read(1)  # Only 1 embedding written, index 1 is out of bounds
        finally:
            os.unlink(tmp_path)

    def test_persistence(self):
        """Test that data persists when reopening the file."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Write data
            service1 = MemoryMappingService(tmp_path, dim=4, capacity=10)
            embeddings = [
                np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32),
            ]
            indices = [service1.write(emb) for emb in embeddings]
            del service1  # Close the file
            
            # Reopen and read
            # Note: size is not persisted, so we need to manually track it
            # or read directly from file indices
            service2 = MemoryMappingService(tmp_path, dim=4, capacity=10)
            # Data is persisted in the file, but size counter resets
            # We can still read the data directly from the file
            np.testing.assert_array_equal(service2.file[0], embeddings[0])
            np.testing.assert_array_equal(service2.file[1], embeddings[1])
            # Size resets to 0 when reopening
            assert service2.size == 0
        finally:
            os.unlink(tmp_path)


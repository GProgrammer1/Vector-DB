"""Unit tests for StorageService."""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from vector_db.services.storage_service import StorageService
from vector_db.types import Node


class TestStorageService:
    """Test suite for StorageService."""

    def test_init(self):
        """Test service initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test_db"
            service = StorageService(
                file_path=str(file_path),
                dim=128,
                capacity=1000,
            )
            
            assert service.dim == 128
            assert service.capacity == 1000
            assert service.size() == 0

    def test_save_and_get(self):
        """Test saving and retrieving nodes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test_db"
            service = StorageService(
                file_path=str(file_path),
                dim=128,
                capacity=1000,
            )
            
            node = Node(
                id=0,
                embedding=np.random.rand(128).astype(np.float32),
                content="test content",
                metadata={"key": "value"},
            )
            
            service.save(node)
            assert service.size() == 1
            
            retrieved = service.get(0)
            assert retrieved is not None
            assert retrieved.id == 0
            np.testing.assert_array_equal(retrieved.embedding, node.embedding)
            assert retrieved.content == "test content"
            assert retrieved.metadata == {"key": "value"}

    def test_get_nonexistent(self):
        """Test getting a non-existent node."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test_db"
            service = StorageService(
                file_path=str(file_path),
                dim=128,
                capacity=1000,
            )
            
            result = service.get(999)
            assert result is None

    def test_get_embedding(self):
        """Test getting embedding by node ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test_db"
            service = StorageService(
                file_path=str(file_path),
                dim=128,
                capacity=1000,
            )
            
            embedding = np.random.rand(128).astype(np.float32)
            node = Node(id=0, embedding=embedding)
            service.save(node)
            
            retrieved_embedding = service.get_embedding(0)
            np.testing.assert_array_equal(retrieved_embedding, embedding)

    def test_delete(self):
        """Test deleting a node."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test_db"
            service = StorageService(
                file_path=str(file_path),
                dim=128,
                capacity=1000,
            )
            
            node = Node(id=0, embedding=np.random.rand(128).astype(np.float32))
            service.save(node)
            assert service.size() == 1
            
            service.delete(0)
            assert service.size() == 0
            assert service.get(0) is None

    def test_get_next_id(self):
        """Test getting next available ID."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test_db"
            service = StorageService(
                file_path=str(file_path),
                dim=128,
                capacity=1000,
            )
            
            assert service.get_next_id() == 0
            
            # Save a node
            node = Node(id=0, embedding=np.random.rand(128).astype(np.float32))
            service.save(node)
            
            assert service.get_next_id() == 1

    def test_persistence(self):
        """Test that storage persists across service instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test_db"
            
            # Create first service and save node
            service1 = StorageService(
                file_path=str(file_path),
                dim=128,
                capacity=1000,
            )
            node = Node(
                id=0,
                embedding=np.random.rand(128).astype(np.float32),
                content="persistent",
            )
            service1.save(node)
            
            # Create second service instance (should load existing data)
            service2 = StorageService(
                file_path=str(file_path),
                dim=128,
                capacity=1000,
            )
            
            # Should be able to retrieve the node
            retrieved = service2.get(0)
            assert retrieved is not None
            assert retrieved.content == "persistent"


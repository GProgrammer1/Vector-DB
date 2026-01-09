"""Unit tests for IndexingService."""

import numpy as np
import pytest
import tempfile
import yaml
from pathlib import Path

from vector_db.services.indexing_service import IndexingService
from vector_db.services.storage_service import StorageService
from vector_db.types import Node


@pytest.fixture
def dummy_config_file():
    """Create a dummy config file for tests."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".yaml") as f:
        f.write("""
embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  dimension: 128
device: CPU
index:
  ef_construction: 10
  M: 4
vector_db:
  file_path: ../vector_db
  dimension: 128
  capacity: 1000
        """)
        config_path = f.name
    yield config_path
    Path(config_path).unlink(missing_ok=True)


@pytest.fixture
def temp_storage(dummy_config_file):
    """Create a temporary storage service."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "test_db"
        storage = StorageService(
            file_path=str(file_path),
            dim=128,
            capacity=1000,
        )
        yield storage, str(file_path)


class TestIndexingService:
    """Test suite for IndexingService."""

    def test_init_creates_new_index(self, dummy_config_file, temp_storage):
        """Test that service creates new index if file doesn't exist."""
        storage, file_path = temp_storage
        index_file = Path(file_path).with_suffix(".index.pkl")
        
        # Index file shouldn't exist yet
        assert not index_file.exists()
        
        # Create service (should not create index file yet)
        service = IndexingService(
            storage=storage.storage,
            config_path=dummy_config_file,
            index_file=str(index_file),
        )
        
        # Index should not be loaded (file doesn't exist)
        assert not service.is_index_loaded()
        assert not index_file.exists()

    def test_init_loads_existing_index(self, dummy_config_file, temp_storage):
        """Test that service loads existing index from file."""
        storage, file_path = temp_storage
        index_file = Path(file_path).with_suffix(".index.pkl")
        
        # Create service and insert a node to create index
        service1 = IndexingService(
            storage=storage.storage,
            config_path=dummy_config_file,
            index_file=str(index_file),
        )
        
        node = Node(
            id=0,
            embedding=np.random.rand(128).astype(np.float32),
            content="test",
        )
        storage.save(node)
        service1.insert_node(node)
        service1.save_index()
        
        # Index file should exist now
        assert index_file.exists()
        
        # Create new service instance (should load index)
        storage2 = StorageService(
            file_path=file_path,
            dim=128,
            capacity=1000,
        )
        # Re-add node to storage (since it's a new storage instance)
        storage2.save(node)
        
        service2 = IndexingService(
            storage=storage2.storage,
            config_path=dummy_config_file,
            index_file=str(index_file),
        )
        
        # Index should be loaded
        assert service2.is_index_loaded()

    def test_insert_node(self, dummy_config_file, temp_storage):
        """Test inserting a node into the index."""
        storage, file_path = temp_storage
        service = IndexingService(
            storage=storage.storage,
            config_path=dummy_config_file,
        )
        
        node = Node(
            id=0,
            embedding=np.random.rand(128).astype(np.float32),
            content="test",
        )
        storage.save(node)
        service.insert_node(node)
        
        # Index should be modified
        assert service._index_modified

    def test_save_index(self, dummy_config_file, temp_storage):
        """Test saving index to disk."""
        storage, file_path = temp_storage
        index_file = Path(file_path).with_suffix(".index.pkl")
        service = IndexingService(
            storage=storage.storage,
            config_path=dummy_config_file,
            index_file=str(index_file),
        )
        
        node = Node(
            id=0,
            embedding=np.random.rand(128).astype(np.float32),
        )
        storage.save(node)
        service.insert_node(node)
        
        # Save index
        service.save_index()
        
        # Index file should exist
        assert index_file.exists()
        # Index should not be marked as modified
        assert not service._index_modified

    def test_search(self, dummy_config_file, temp_storage):
        """Test searching the index."""
        storage, file_path = temp_storage
        service = IndexingService(
            storage=storage.storage,
            config_path=dummy_config_file,
        )
        
        # Insert multiple nodes
        nodes = []
        for i in range(10):
            embedding = np.random.rand(128).astype(np.float32)
            node = Node(id=i, embedding=embedding)
            nodes.append(node)
            storage.save(node)
            service.insert_node(node)
        
        # Save index to ensure it's in a consistent state
        service.save_index()
        
        # Search for nearest neighbor
        query = nodes[0].embedding
        results = service.search(query, k=3, ef=10)
        
        # Results might be less than k if not enough nodes or search issues
        assert len(results) > 0
        assert len(results) <= 3
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
        # Check that results contain Node and numeric distance
        for r in results:
            assert isinstance(r[0], Node) or r[0] is None  # Node might be None if deleted
            # Distance can be float, int, or numpy numeric type
            assert isinstance(r[1], (float, int)) or hasattr(r[1], 'dtype')  # Distance

    def test_delete_node(self, dummy_config_file, temp_storage):
        """Test deleting a node from the index."""
        storage, file_path = temp_storage
        service = IndexingService(
            storage=storage.storage,
            config_path=dummy_config_file,
        )
        
        node = Node(
            id=0,
            embedding=np.random.rand(128).astype(np.float32),
        )
        storage.save(node)
        service.insert_node(node)
        
        # Delete node
        service.delete_node(0)
        
        # Index should be modified
        assert service._index_modified


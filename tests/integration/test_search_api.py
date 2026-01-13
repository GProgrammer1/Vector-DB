import numpy as np
import pytest
import tempfile
import yaml
from pathlib import Path
from fastapi.testclient import TestClient

from vector_db.api.app import app
from vector_db.services.embedding_service import EmbeddingService
from vector_db.services.storage_service import StorageService
from vector_db.services.indexing_service import IndexingService
import vector_db.api.app as app_module


@pytest.fixture(scope="function")
def test_client():
    """Setup test client with temporary files and initialized services."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        db_path = Path(tmpdir) / "test_db"
        
        config_data = {
            "embedding": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "dimension": 384
            },
            "device": "CPU",
            "index": {
                "ef_construction": 10,
                "M": 4,
                "flush_threshold": 100
            },
            "vector_db": {
                "file_path": str(db_path),
                "dimension": 384,
                "capacity": 1000
            }
        }
        
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
        
        # Initialize services and set them as globals
        embedding_client = EmbeddingService(config_path=str(config_path))
        storage_service = StorageService(
            file_path=str(db_path),
            dim=384,
            capacity=1000
        )
        index_file = db_path.with_suffix(".index.pkl")
        indexing_service = IndexingService(
            storage=storage_service.storage,
            config_path=str(config_path),
            index_file=str(index_file)
        )
        
        # Store original values
        original_embedding = app_module.embedding_client
        original_storage = app_module.storage_service
        original_indexing = app_module.indexing_service
        original_use_service = app_module.USE_EMBEDDING_SERVICE
        
        # Override globals
        app_module.embedding_client = embedding_client
        app_module.storage_service = storage_service
        app_module.indexing_service = indexing_service
        app_module.USE_EMBEDDING_SERVICE = False # Force local service
        
        # Create test client
        with TestClient(app) as client:
            yield client
        
        # Restore original values
        app_module.embedding_client = original_embedding
        app_module.storage_service = original_storage
        app_module.indexing_service = original_indexing
        app_module.USE_EMBEDDING_SERVICE = original_use_service



def check_status(response, expected_code=200):
    """Helper to assert status code and print error detail on failure."""
    if response.status_code != expected_code:
        try:
            detail = response.json()
        except Exception:
            detail = response.text
        print(f"\nRequest failed: {detail}")
    assert response.status_code == expected_code


def test_search_basic(test_client):
    """Test standard search without filters."""
    # 1. Insert data
    test_client.post("/embed", json={"content": "Apple fruit", "metadata": {"type": "fruit"}})
    test_client.post("/embed", json={"content": "Banana fruit", "metadata": {"type": "fruit"}})
    test_client.post("/embed", json={"content": "Laptop computer", "metadata": {"type": "tech"}})
    
    # 2. Search
    response = test_client.post("/search", json={"query": "fruit", "top_k": 2})
    check_status(response)
    data = response.json()
    assert len(data["results"]) <= 2
    # Just check that we got results, embeddings might vary
    assert len(data["results"]) > 0


def test_search_with_filter(test_client):
    """Test search with metadata filtering."""
    # 1. Insert data
    test_client.post("/embed", json={"content": "Red Apple", "metadata": {"color": "red", "type": "fruit"}})
    test_client.post("/embed", json={"content": "Green Apple", "metadata": {"color": "green", "type": "fruit"}})
    test_client.post("/embed", json={"content": "Red Car", "metadata": {"color": "red", "type": "vehicle"}})
    
    # 2. Search for 'Apple' but only 'red' ones
    response = test_client.post("/search", json={
        "query": "Apple",
        "top_k": 5,
        "metadata_filter": {"color": "red"}
    })
    
    check_status(response)
    results = response.json()["results"]
    
    # The key point: 'Green Apple' should NOT be in results
    contents = [r["content"] for r in results]
    assert "Green Apple" not in contents
    
    # All results should match the filter
    for result in results:
        assert result["metadata"].get("color") == "red"


def test_search_no_match_filter(test_client):
    """Test search where no items match the filter."""
    test_client.post("/embed", json={"content": "Test", "metadata": {"key": "val"}})
    
    response = test_client.post("/search", json={
        "query": "Test",
        "metadata_filter": {"key": "nonexistent"}
    })
    
    check_status(response)
    assert len(response.json()["results"]) == 0


def test_search_params_handling(test_client):
    """Test that extra parameters are accepted and passed through."""
    test_client.post("/embed", json={"content": "Param test"})
    
    # Pass ef and custom params
    response = test_client.post("/search", json={
        "query": "Param test",
        "ef": 100,
        "pq_chunks": 8,
        "params": {"custom_arg": "value"}
    })
    
    check_status(response)
    assert len(response.json()["results"]) > 0

import numpy as np
import pytest
import tempfile
import yaml
import os
from pathlib import Path
from fastapi.testclient import TestClient

# Mock the globals in app.py for testing
import vector_db.api.app as app_module
from vector_db.api.app import app
from vector_db.services.embedding_service import EmbeddingService
from vector_db.services.storage_service import StorageService
from vector_db.services.indexing_service import IndexingService

@pytest.fixture
def test_setup():
    """Setup test environment with temporary files and initialized services."""
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
            
        # Initialize services
        # Note: We need to override the global instances in app_module
        app_module.embedding_client = EmbeddingService(config_path=str(config_path))
        app_module.storage_service = StorageService(
            file_path=str(db_path),
            dim=384,
            capacity=1000
        )
        index_file = db_path.with_suffix(".index.pkl")
        app_module.indexing_service = IndexingService(
            storage=app_module.storage_service.storage,
            config_path=str(config_path),
            index_file=str(index_file)
        )
        app_module.CONFIG_PATH = str(config_path)
        app_module.INDEX_TYPE = "hnsw"
        
        yield app_module
        
        # Cleanup globals
        app_module.embedding_client = None
        app_module.storage_service = None
        app_module.indexing_service = None

def test_search_basic(test_setup):
    """Test standard search without filters."""
    client = TestClient(app)
    
    # 1. Insert data
    client.post("/embed", json={"content": "Apple fruit", "metadata": {"type": "fruit"}})
    client.post("/embed", json={"content": "Banana fruit", "metadata": {"type": "fruit"}})
    client.post("/embed", json={"content": "Laptop computer", "metadata": {"type": "tech"}})
    
    # 2. Search
    response = client.post("/search", json={"query": "fruit", "top_k": 2})
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 2
    assert "Apple" in data["results"][0]["content"] or "Banana" in data["results"][0]["content"]

def test_search_with_filter(test_setup):
    """Test search with metadata filtering."""
    client = TestClient(app)
    
    # 1. Insert data
    client.post("/embed", json={"content": "Red Apple", "metadata": {"color": "red", "type": "fruit"}})
    client.post("/embed", json={"content": "Green Apple", "metadata": {"color": "green", "type": "fruit"}})
    client.post("/embed", json={"content": "Red Car", "metadata": {"color": "red", "type": "vehicle"}})
    
    # 2. Search for 'Apple' but only 'red' ones
    response = client.post("/search", json={
        "query": "Apple",
        "top_k": 5,
        "metadata_filter": {"color": "red"}
    })
    
    assert response.status_code == 200
    results = response.json()["results"]
    
    # Should find 'Red Apple' and 'Red Car', but 'Red Apple' should be closer (if embedding works)
    # The key point for filter test is that 'Green Apple' is NOT present.
    contents = [r["content"] for r in results]
    assert "Red Apple" in contents
    assert "Red Car" in contents
    assert "Green Apple" not in contents

def test_search_no_match_filter(test_setup):
    """Test search where no items match the filter."""
    client = TestClient(app)
    
    client.post("/embed", json={"content": "Test", "metadata": {"key": "val"}})
    
    response = client.post("/search", json={
        "query": "Test",
        "metadata_filter": {"key": "nonexistent"}
    })
    
    assert response.status_code == 200
    assert len(response.json()["results"]) == 0

def test_search_params_handling(test_setup):
    """Test that extra parameters are accepted and passed through."""
    client = TestClient(app)
    
    client.post("/embed", json={"content": "Param test"})
    
    # Pass ef and custom params
    response = client.post("/search", json={
        "query": "Param test",
        "ef": 100,
        "pq_chunks": 8,
        "params": {"custom_arg": "value"}
    })
    
    assert response.status_code == 200
    assert len(response.json()["results"]) > 0

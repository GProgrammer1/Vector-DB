"""Integration tests for the /embed API endpoint."""

import numpy as np
import pytest
import tempfile
import yaml
from pathlib import Path

try:
    from fastapi.testclient import TestClient
    from vector_db.api.app import app, embedding_service, storage_service, indexing_service
    from vector_db.services.embedding_service import EmbeddingService
    from vector_db.services.storage_service import StorageService
    from vector_db.services.indexing_service import IndexingService
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False
    TestClient = None  # type: ignore[assignment, misc]

pytestmark = pytest.mark.skipif(
    not _FASTAPI_AVAILABLE,
    reason="fastapi not installed"
)


@pytest.fixture
def dummy_config_file():
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".yaml") as f:
        f.write("""
embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  dimension: 384
device: CPU
index:
  ef_construction: 10
  M: 4
vector_db:
  file_path: ../vector_db
  dimension: 384
  capacity: 1000
        """)
        config_path = f.name
    yield config_path
    Path(config_path).unlink(missing_ok=True)


@pytest.fixture
def temp_data_dir(dummy_config_file):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Update config to use temp directory
        with open(dummy_config_file, 'r') as f:
            config = yaml.safe_load(f)
        config['vector_db']['file_path'] = str(Path(tmpdir) / "test_db")
        with open(dummy_config_file, 'w') as f:
            yaml.dump(config, f)
        
        yield tmpdir, dummy_config_file


@pytest.fixture
def initialized_app(temp_data_dir):
    tmpdir, config_path = temp_data_dir
    
    # Initialize services
    global embedding_service, storage_service, indexing_service
    
    embedding_service = EmbeddingService(config_path=config_path)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    vector_db_config = config.get("vector_db", {})
    
    storage_service = StorageService(
        file_path=vector_db_config.get("file_path"),
        dim=384,
        capacity=1000,
    )
    
    index_file = Path(vector_db_config.get("file_path")).with_suffix(".index.pkl")
    indexing_service = IndexingService(
        storage=storage_service.storage,
        config_path=config_path,
        index_file=str(index_file),
    )
    
    yield app
    
    # Cleanup
    if indexing_service:
        indexing_service.save_index()


class TestEmbedAPI:

    def test_embed_first_document(self, initialized_app, temp_data_dir):
        client = TestClient(initialized_app)
        
        response = client.post(
            "/embed",
            json={
                "content": "This is a test document",
                "metadata": {"source": "test"}
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status_code"] == 200
        assert "successfully" in data["message"].lower()
        assert data["error"] is None
        assert "index" in data["message"]

    def test_embed_multiple_documents(self, initialized_app, temp_data_dir):
        client = TestClient(initialized_app)
        
        # First document
        response1 = client.post(
            "/embed",
            json={"content": "First document"}
        )
        assert response1.status_code == 200
        
        # Second document
        response2 = client.post(
            "/embed",
            json={"content": "Second document"}
        )
        assert response2.status_code == 200
        
        # Third document
        response3 = client.post(
            "/embed",
            json={"content": "Third document", "metadata": {"id": 3}}
        )
        assert response3.status_code == 200
        
        assert response1.json()["status_code"] == 200
        assert response2.json()["status_code"] == 200
        assert response3.json()["status_code"] == 200

    def test_embed_with_metadata(self, initialized_app, temp_data_dir):
        client = TestClient(initialized_app)
        
        response = client.post(
            "/embed",
            json={
                "content": "Document with metadata",
                "metadata": {
                    "author": "test",
                    "category": "test",
                    "tags": ["tag1", "tag2"]
                }
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status_code"] == 200

    def test_embed_empty_content(self, initialized_app, temp_data_dir):
        client = TestClient(initialized_app)
        
        response = client.post(
            "/embed",
            json={"content": ""}
        )
        
        assert response.status_code == 200

    def test_health_endpoint(self, initialized_app, temp_data_dir):
        """Test health check endpoint."""
        client = TestClient(initialized_app)
        
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "indexing"
        assert "index_loaded" in data
        assert "storage_size" in data

    def test_index_persistence(self, initialized_app, temp_data_dir):
        client = TestClient(initialized_app)
        tmpdir, config_path = temp_data_dir
        
        # Insert first document
        response1 = client.post(
            "/embed",
            json={"content": "Persistent document"}
        )
        assert response1.status_code == 200
        
        # Check that index file was created
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        file_path = config['vector_db']['file_path']
        index_file = Path(file_path).with_suffix(".index.pkl")
        
        # Index should exist after first insert
        assert index_file.exists()


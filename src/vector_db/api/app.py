import numpy as np
import yaml
import os
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, Union
from fastapi import FastAPI, HTTPException

from .models import InsertRequest, InsertResponse
from ..services.storage_service import StorageService
from ..services.indexing_service import IndexingService
from ..services.embedding_client import SyncEmbeddingClient
from ..types import Node

# Load config path
CONFIG_PATH = os.getenv("CONFIG_PATH", str(Path(__file__).parent.parent.parent / "config.yaml"))

# Global service instances (initialized on startup)
# embedding_client can be either SyncEmbeddingClient (HTTP) or EmbeddingService (local)
from typing import Union
try:
    from ..services.embedding_service import EmbeddingService as LocalEmbeddingService
    EmbeddingClientType = Union[SyncEmbeddingClient, LocalEmbeddingService]
except ImportError:
    EmbeddingClientType = SyncEmbeddingClient  # type: ignore[misc,assignment]

embedding_client: Optional[EmbeddingClientType] = None
storage_service: Optional[StorageService] = None
indexing_service: Optional[IndexingService] = None

# Configuration: use embedding service API or local embedding service
USE_EMBEDDING_SERVICE = os.getenv("USE_EMBEDDING_SERVICE", "true").lower() == "true"
EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://embedding-service:8001")


def load_config() -> dict:
    """Load configuration from YAML file."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage service lifecycle: initialize on startup, cleanup on shutdown."""
    global embedding_client, storage_service, indexing_service
    
    config = load_config()
    vector_db_config = config.get("vector_db", {})
    embedding_config = config.get("embedding", {})
    
    file_path = vector_db_config.get("file_path", "../vector_db")
    capacity = vector_db_config.get("capacity", 1000000)
    dim = embedding_config.get("dimension", 384)
    
    print(f"Initializing services with config at {CONFIG_PATH}...")
    
    if USE_EMBEDDING_SERVICE:
        embedding_client = SyncEmbeddingClient(base_url=EMBEDDING_SERVICE_URL)
        if embedding_client.health_check():
            print(f"Embedding service client initialized (URL: {EMBEDDING_SERVICE_URL})")
        else:
            print(f"Warning: Embedding service at {EMBEDDING_SERVICE_URL} is not healthy")
    else:
        # Fallback to local embedding service (for development/testing)
        try:
            from ..services.embedding_service import EmbeddingService as LocalEmbeddingService
            embedding_client = LocalEmbeddingService(config_path=str(CONFIG_PATH))
            print(f"Local embedding service initialized (dim={embedding_client.dim})")
        except ImportError:
            raise RuntimeError(
                "Cannot use local embedding service: sentence-transformers not installed. "
                "Set USE_EMBEDDING_SERVICE=true to use HTTP API."
            )
    
    storage_service = StorageService(
        file_path=file_path,
        dim=dim,
        capacity=capacity,
    )
    print(f"Storage service initialized (size={storage_service.size()})")
    
    index_file = Path(file_path).with_suffix(".index.pkl")
    indexing_service = IndexingService(
        storage=storage_service.storage,
        config_path=str(CONFIG_PATH),
        index_file=str(index_file),
    )
    
    if indexing_service.is_index_loaded():
        print(f"Index loaded from {index_file} (size={storage_service.size()} nodes)")
    else:
        print(f"Index not found at {index_file}, will create on first insert")
    
    yield
    
    if indexing_service:
        indexing_service.save_index()
        print(f"Index saved to disk on shutdown (size={indexing_service.get_index_size()} nodes)")
    
    if embedding_client and hasattr(embedding_client, 'close'):
        embedding_client.close()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "index_loaded": indexing_service.is_index_loaded() if indexing_service else False,
        "index_size": indexing_service.get_index_size() if indexing_service else 0,
        "storage_size": storage_service.size() if storage_service else 0,
        "index_modified": indexing_service._index_modified if indexing_service else False,
    }


@app.post("/embed", response_model=InsertResponse)
def embed_document(insert_request: InsertRequest):
    """
    Embed a document and store it in the vector database.
    """
    global embedding_client, storage_service, indexing_service
    
    if embedding_client is None or storage_service is None or indexing_service is None:
        raise HTTPException(
            status_code=503,
            detail="Services not initialized"
        )
    
    try:
        content = insert_request.content
        embedding = embedding_client.embed_text(content)
        
        # Create node
        node_id = storage_service.get_next_id()
        node = Node(
            id=node_id,
            embedding=embedding,
            content=content,
            metadata=insert_request.metadata or {},
        )
        
        # Save to storage
        storage_service.save(node)
        
        # Add to index
        indexing_service.insert_node(node)
        
        return InsertResponse(
            status_code=200,
            message=f"Document embedded and stored successfully at index {node_id}",
            error=None
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )


from .models import InsertRequest, InsertResponse, QueryRequest, QueryResponse

@app.post("/search", response_model=QueryResponse)
def search_index(query_request: QueryRequest):
    """
    Search for nearest neighbors in the index with optional metadata filtering.
    """
    global embedding_client, storage_service, indexing_service
    
    if embedding_client is None or storage_service is None or indexing_service is None:
        raise HTTPException(
            status_code=503,
            detail="Services not initialized"
        )
    
    try:
        # 1. Generate embedding for query
        query_embedding = embedding_client.embed_text(query_request.query)
        
        # 2. Handle metadata filtering
        filter_ids = None
        if query_request.metadata_filter:
            filter_ids = storage_service.filter_by_metadata(query_request.metadata_filter)
            if not filter_ids:
                # If filter returned nothing, no need to search index
                return QueryResponse(
                    status_code=200,
                    results=[],
                    error=None
                )
        
        # 3. Search index
        # Collect search arguments
        search_kwargs = {
            "ef": query_request.ef,
            "filter_ids": filter_ids
        }
        
        # Add optional/generic parameters
        if query_request.pq_chunks:
            search_kwargs["pq_chunks"] = query_request.pq_chunks
        if query_request.params:
            search_kwargs.update(query_request.params)
            
        results = indexing_service.search(
            query=query_embedding,
            k=query_request.top_k,
            **search_kwargs
        )
        
        # 4. Format results
        formatted_results = []
        for node, dist in results:
            formatted_results.append({
                "id": node.id,
                "content": node.content,
                "metadata": node.metadata,
                "distance": float(dist)
            })
            
        return QueryResponse(
            status_code=200,
            results=formatted_results,
            error=None
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing search: {str(e)}"
        )

    


import numpy as np
import yaml
from pathlib import Path
from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer

from .models import InsertRequest, InsertResponse
from ..inference.embedding import EmbeddingService
from ..inference.mmap_vector_store import MemoryMappingService

app = FastAPI()

# Load config once at startup
CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"


def load_config():
    """Load configuration from YAML file."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


@app.post("/embed", response_model=InsertResponse)
def embed_document(insert_request: InsertRequest):
    """Embed a document and store it in the vector database."""
    try:
        # Load config
        config = load_config()
        embedding_config = config.get("embedding", {})
        vector_db_config = config.get("vector_db", {})
        
        # Get embedding configuration
        model_name = embedding_config.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        device = embedding_config.get("device", "auto").lower()
        dim = embedding_config.get("dimension", 384)
        
        # Get vector_db configuration
        file_path = vector_db_config.get("file_path", "../vector_db")
        capacity = vector_db_config.get("capacity", 1000000)
        
        # Initialize embedding service
        model = SentenceTransformer(model_name)
        embedder = EmbeddingService(model, device=device)
        
        # Embed text
        content = insert_request.content
        embedding = embedder.embed_text(content)
        
        # Ensure embedding is float32 and correct dimension
        embedding = embedding.astype(np.float32)
        if embedding.shape[0] != dim:
            raise ValueError(f"Embedding dimension mismatch: expected {dim}, got {embedding.shape[0]}")
        
        # Initialize vector store
        vector_store = MemoryMappingService(
            file_path=file_path,
            dim=dim,
            capacity=capacity,
            config_path=str(CONFIG_PATH)
        )
        
        # Write embedding to store
        idx = vector_store.write(embedding)
        
        # Add to HNSW index
        vector_store.index.insert_node(node_id=idx, embedding=embedding)
        
        return InsertResponse(
            status_code=200,
            message=f"Document embedded and stored successfully at index {idx}",
            error=None
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )

    


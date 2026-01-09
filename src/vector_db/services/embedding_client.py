"""HTTP client for embedding service API."""

import numpy as np
from typing import List, Optional
import httpx
from pathlib import Path


class EmbeddingClient:
    """
    HTTP client for communicating with the embedding service API.
    
    This allows the indexing service to be decoupled from the embedding service,
    enabling independent scaling of GPU-intensive embedding generation.
    """

    def __init__(
        self,
        base_url: str = "http://embedding-service:8001",
        timeout: float = 30.0,
    ):
        """
        Initialize the embedding client.
        
        Args:
            base_url: Base URL of the embedding service
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    async def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text via HTTP.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        response = await self._client.post(
            f"{self.base_url}/embed",
            json={"text": text}
        )
        response.raise_for_status()
        data = response.json()
        return np.array(data["embedding"], dtype=np.float32)

    async def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed multiple texts via HTTP.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Array of embedding vectors
        """
        response = await self._client.post(
            f"{self.base_url}/embed/batch",
            json={"texts": texts}
        )
        response.raise_for_status()
        data = response.json()
        embeddings = np.array(data["embeddings"], dtype=np.float32)
        return embeddings

    async def health_check(self) -> bool:
        """
        Check if embedding service is healthy.
        
        Returns:
            True if service is healthy, False otherwise
        """
        try:
            response = await self._client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()


class SyncEmbeddingClient:
    """
    Synchronous HTTP client for embedding service API.
    
    For use in synchronous contexts (like FastAPI endpoints).
    """

    def __init__(
        self,
        base_url: str = "http://embedding-service:8001",
        timeout: float = 30.0,
    ):
        """
        Initialize the synchronous embedding client.
        
        Args:
            base_url: Base URL of the embedding service
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text via HTTP.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        response = self._client.post(
            f"{self.base_url}/embed",
            json={"text": text}
        )
        response.raise_for_status()
        data = response.json()
        return np.array(data["embedding"], dtype=np.float32)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed multiple texts via HTTP.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Array of embedding vectors
        """
        response = self._client.post(
            f"{self.base_url}/embed/batch",
            json={"texts": texts}
        )
        response.raise_for_status()
        data = response.json()
        embeddings = np.array(data["embeddings"], dtype=np.float32)
        return embeddings

    def health_check(self) -> bool:
        """
        Check if embedding service is healthy.
        
        Returns:
            True if service is healthy, False otherwise
        """
        try:
            response = self._client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False

    def close(self):
        """Close the HTTP client."""
        self._client.close()


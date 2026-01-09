"""Embedding service wrapper for managing embedding model lifecycle."""

import numpy as np
from typing import Optional, List, TYPE_CHECKING
from pathlib import Path
import yaml

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

try:
    from sentence_transformers import SentenceTransformer
    from ..inference.embedding import EmbeddingService as BaseEmbeddingService
    from ..inference.device import DeviceType, get_device
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None  # type: ignore[assignment, misc]
    BaseEmbeddingService = None  # type: ignore[assignment, misc]
    DeviceType = str  # type: ignore[assignment, misc]


class EmbeddingService:
    """
    Service for managing embedding model lifecycle.
    
    Handles:
    - Loading model from config
    - Device management
    - Batch processing
    """

    def __init__(
        self,
        config_path: str,
        model: Optional["SentenceTransformer"] = None,
    ):
        """
        Initialize the embedding service.
        
        Args:
            config_path: Path to config.yaml
            model: Optional pre-loaded model (for testing or reuse)
        """
        if not _SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install it with: pip install sentence-transformers"
            )
        
        self.config_path = Path(config_path)
        
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
        
        embedding_config = config.get("embedding", {})
        device_config = config.get("device", "auto")
        
        # Get model name
        model_name = embedding_config.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        self.dimension = embedding_config.get("dimension", 384)
        
        # Initialize model if not provided
        if model is None:
            model = SentenceTransformer(model_name)  # type: ignore[misc]
        
        # Initialize base embedding service
        # Convert device_config to DeviceType (handle string to Literal conversion)
        if isinstance(device_config, str):
            device_str = device_config.lower()
            # Type narrowing for DeviceType
            if device_str in ("cpu", "cuda", "mps", "auto"):
                device: DeviceType = device_str  # type: ignore[assignment]
            else:
                device = "auto"  # type: ignore[assignment]
        else:
            device = "auto"  # type: ignore[assignment]
        self._service = BaseEmbeddingService(model, device=device)  # type: ignore[misc]

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text string.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        embedding = self._service.embed_text(text)
        # Ensure correct dtype and dimension
        embedding = embedding.astype(np.float32)
        if embedding.shape[0] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, got {embedding.shape[0]}"
            )
        return embedding

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple text strings.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Array of embedding vectors
        """
        embeddings = self._service.embed_texts(texts)
        # Ensure correct dtype
        embeddings = embeddings.astype(np.float32)
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, got {embeddings.shape[1]}"
            )
        return embeddings

    @property
    def dim(self) -> int:
        """Get embedding dimension."""
        return self.dimension


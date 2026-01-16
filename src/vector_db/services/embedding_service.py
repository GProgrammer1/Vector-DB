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
        if not _SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is not installed. "
            )
        
        self.config_path = Path(config_path)
        
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
        
        if "embedding" not in config:
            raise ValueError("Config file must contain 'embedding' section")
        embedding_config = config["embedding"]
        
        if "device" not in config:
            raise ValueError("Config file must contain 'device' setting")
        device_config = config["device"]
        
        if "model" not in embedding_config:
            raise ValueError("Config 'embedding.model' is required")
        model_name = embedding_config["model"]
        
        if "dimension" not in embedding_config:
            raise ValueError("Config 'embedding.dimension' is required")
        self.dimension = embedding_config["dimension"]
        
        if model is None:
            model = SentenceTransformer(model_name)  # type: ignore[misc]
        if isinstance(device_config, str):
            device_str = device_config.lower()
            if device_str in ("cpu", "cuda", "mps", "auto"):
                device: DeviceType = device_str  # type: ignore[assignment]
            else:
                device = "auto"  # type: ignore[assignment]
        else:
            device = "auto"  # type: ignore[assignment]
        self._service = BaseEmbeddingService(model, device=device)  # type: ignore[misc]

    def embed_text(self, text: str) -> np.ndarray:
    
        embedding = self._service.embed_text(text)
        embedding = embedding.astype(np.float32)
        if embedding.shape[0] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, got {embedding.shape[0]}"
            )
        return embedding

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        
        embeddings = self._service.embed_texts(texts)
        embeddings = embeddings.astype(np.float32)
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.dimension}, got {embeddings.shape[1]}"
            )
        return embeddings

    @property
    def dim(self) -> int:
        return self.dimension


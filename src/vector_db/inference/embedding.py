import numpy as np
from typing import Optional

from sentence_transformers import SentenceTransformer
from vector_db.inference.device import DeviceType, get_device


class EmbeddingService:
    """Service for generating embeddings from text using sentence transformers."""

    def __init__(
        self,
        model: SentenceTransformer,
        device: Optional[DeviceType] = "auto",
    ):
        """
        Initialize the embedding service.

        Args:
            model: SentenceTransformer model instance
            device: Device to use for inference. Options:
                - "auto": Automatically select best available device (CPU/GPU)
                - "cpu": Force CPU usage
                - "cuda": Use CUDA GPU if available
                - "mps": Use Apple Silicon GPU if available

        Raises:
            ValueError: If model is None
        """
        if model is None:
            raise ValueError("Model is required")

        self.model = model
        self.device = get_device(device)
        # Move model to the selected device
        self.model = self.model.to(self.device)

    def embed_text(self, text: str) -> np.ndarray:
        
        embedding = self.model.encode(text, device=self.device)
        return embedding

    def embed_texts(self, texts: list[str]) -> np.ndarray:
      
        embeddings = self.model.encode(texts, device=self.device)
        return embeddings


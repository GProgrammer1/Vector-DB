import numpy as np
from typing import Optional, List

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
        """Generate embedding for a single text string."""
        embedding = self.model.encode(text, device=self.device)
        # Convert to numpy array if it's a tensor
        if hasattr(embedding, "numpy"):
            return embedding.numpy()
        return np.asarray(embedding)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple text strings."""
        embeddings = self.model.encode(texts, device=self.device)
        # Convert to numpy array if it's a tensor
        if hasattr(embeddings, "numpy"):
            return embeddings.numpy()
        return np.asarray(embeddings)


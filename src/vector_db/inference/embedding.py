import numpy as np
from typing import Optional, List

from sentence_transformers import SentenceTransformer
from vector_db.inference.device import DeviceType, get_device


class EmbeddingService:

    def __init__(
        self,
        model: SentenceTransformer,
        device: Optional[DeviceType] = "auto",
    ):
        
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


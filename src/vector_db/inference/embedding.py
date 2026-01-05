import numpy as np
from typing import Any
from sentence_transformers import SentenceTransformer
class EmbeddingService():

    def __init__(self, model: SentenceTransformer):
        if model is None:
            raise ValueError("Model is required")
        self.model = model


    def embed_text(self, text: str) -> np.ndarray:
        embedding = self.model.encode(text)
        return embedding

    

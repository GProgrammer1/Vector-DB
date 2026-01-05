import numpy as np
from pathlib import Path

class MemoryMappingService:
    """
    Memory mapping service for storing embeddings in a file.
    The file is a numpy array of shape (capacity, dim).
    """

    def __init__(self, file_path: str, dim: int, capacity: int):
        if dim <= 0:
            raise ValueError("Dimension must be greater than 0")
        if capacity <= 0:
            raise ValueError("Capacity must be greater than 0")

        self.file_path = Path(file_path)
        self.dim = dim
        self.capacity = capacity
        self.size = 0

        mode = "r+" if self.file_path.exists() else "w+"

        self.file = np.memmap(
            self.file_path,
            dtype=np.float32,
            mode=mode,
            shape=(capacity, dim)
        )

    def write(self, embedding: np.ndarray) -> int:
        if not isinstance(embedding, np.ndarray):
            raise TypeError("Embedding must be a numpy array")
        if embedding.dtype != np.float32:
            raise TypeError("Embedding must be a float32 array")
        if embedding.ndim != 1:
            raise ValueError("Embedding must be a 1D array")
        if embedding.size != self.dim:
            raise ValueError(f"Embedding must be of dimension {self.dim}")
        if self.size >= self.capacity:
            raise RuntimeError("Memory-mapped file is full")

        idx = self.size
        self.file[idx] = embedding
        self.size += 1

        return idx

    def read(self, idx: int) -> np.ndarray:
        if not isinstance(idx, int):
            raise TypeError("Index must be an integer")
        if idx < 0 or idx >= self.size:
            raise IndexError(f"Index must be between 0 and {self.size - 1}")
        return self.file[idx]

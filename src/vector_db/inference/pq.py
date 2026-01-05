import numpy as np
from typing import Optional
from scipy.cluster.vq import kmeans2
class ProductQuantizationService:

    """
    Product quantization service for compressing embeddings.
    The service is used to compress embeddings into a smaller number of bits.
    The service is used to decompress embeddings into their original dimension.
    k: k-means paramater (number of centroids)
    chunks: number of chunks to split the embedding into
    dim: dimension of the embedding
    """
    def __init__(self, k: int, chunks: int, dim: int):
        if k <= 0:
            raise ValueError("k must be greater than 0")
        if chunks <= 0:
            raise ValueError("chunks must be greater than 0")
        if dim <= 0:
            raise ValueError("dim must be greater than 0")
        if dim % chunks != 0:
            raise ValueError("dim must be divisible by chunks")

        self.k = k
        self.chunks = chunks
        self.dim = dim
        self.subdim = dim // chunks
        self.centroids: Optional[list[np.ndarray]] = None

    def _validate_embeddings(self, embeddings: np.ndarray) -> None:
        """Validate input embeddings."""
        if not isinstance(embeddings, np.ndarray):
            raise TypeError("Embeddings must be a numpy array")
        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings must be 2D array, got {embeddings.ndim}D")
        if embeddings.shape[1] != self.dim:
            raise ValueError(f"Embedding dimension must be {self.dim}, got {embeddings.shape[1]}")

    def _chunk_embeddings(self, embeddings: np.ndarray) -> list[list[np.ndarray]]:
        """Split embeddings into chunks and collect chunks by position."""
        chunks: list[list[np.ndarray]] = [[] for _ in range(self.chunks)]
        for embedding in embeddings:
            for i in range(self.chunks):
                chunk = embedding[i * self.subdim:(i + 1) * self.subdim]
                chunks[i].append(chunk)
        return chunks

    def _compute_centroids(self, chunks: list[list[np.ndarray]]) -> list[np.ndarray]:
        """Apply k-means clustering to each chunk to compute centroids."""
        centroids: list[np.ndarray] = []
        for chunk_list in chunks:
            chunk_array = np.array(chunk_list)
            chunk_centroids, _ = kmeans2(chunk_array, self.k, iter=100, minit='points')
            centroids.append(chunk_centroids)
        return centroids

    def _find_nearest_centroid(self, chunk: np.ndarray, centroids: np.ndarray) -> int:
        """Find the index of the nearest centroid for a given chunk."""
        distances = np.linalg.norm(centroids - chunk, axis=1)
        return int(np.argmin(distances))

    def _compress_embedding(self, embedding: np.ndarray, centroids: list[np.ndarray]) -> np.ndarray:
        """Compress a single embedding by finding nearest centroids for each chunk."""
        compressed_indices = []
        for i in range(self.chunks):
            chunk = embedding[i * self.subdim:(i + 1) * self.subdim]
            nearest_centroid_idx = self._find_nearest_centroid(chunk, centroids[i])
            compressed_indices.append(nearest_centroid_idx)
        return np.array(compressed_indices, dtype=np.int64)  # type: ignore[no-any-return]

    def compress(self, embeddings: np.ndarray) -> np.ndarray:
        """Compress embeddings using product quantization."""
        self._validate_embeddings(embeddings)
        
        # Split embeddings into chunks
        chunks = self._chunk_embeddings(embeddings)
        
        # Compute centroids for each chunk
        centroids = self._compute_centroids(chunks)
        self.centroids = centroids
        
        # Compress each embedding
        compressed_embeddings = [
            self._compress_embedding(embedding, centroids)
            for embedding in embeddings
        ]
        
        return np.array(compressed_embeddings, dtype=np.int64)  # type: ignore[no-any-return]

    
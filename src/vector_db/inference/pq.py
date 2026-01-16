import numpy as np
from typing import Optional, List, Tuple
from scipy.cluster.vq import kmeans2
from concurrent.futures import ProcessPoolExecutor


def _kmeans_chunk_worker(args: Tuple[List[np.ndarray], int]) -> np.ndarray:
    """Worker function for parallel k-means computation (must be module-level for pickling)."""
    chunk_list, k = args
    chunk_array = np.array(chunk_list, dtype=np.float32)
    centroids, _ = kmeans2(chunk_array, k, iter=100, minit='points')
    return centroids

class ProductQuantizationService:

    """
    k: k-means parameter (number of centroids)
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
        self.centroids: Optional[List[np.ndarray]] = None

    def _validate_embeddings(self, embeddings: np.ndarray) -> None:
        if not isinstance(embeddings, np.ndarray):
            raise TypeError("Embeddings must be a numpy array")
        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings must be 2D array, got {embeddings.ndim}D")
        if embeddings.shape[1] != self.dim:
            raise ValueError(f"Embedding dimension must be {self.dim}, got {embeddings.shape[1]}")

    def _chunk_embeddings(self, embeddings: np.ndarray) -> List[List[np.ndarray]]:
        N, D = embeddings.shape

        # Ensure embedding dimension matches expected total size
        assert D == self.subdim * self.chunks

        # Reshape embeddings to (N, chunks, subdim) so each embedding is split into `chunks` sub-vectors
        reshaped = embeddings.reshape(N, self.chunks, self.subdim)

        # Reorder axes to (chunks, N, subdim) to group all vectors of the same chunk index together
        chunks = reshaped.transpose(1, 0, 2)

        # Convert each chunk into a list of vectors for more processing
        return [list(chunk) for chunk in chunks]

    def _compute_centroids(self, chunks: List[List[np.ndarray]]) -> List[np.ndarray]:
        """
        Apply k-means clustering to each chunk in parallel across multiple processes.
        """
        args = [(chunk, self.k) for chunk in chunks]

        
        with ProcessPoolExecutor() as executor:
            results = executor.map(_kmeans_chunk_worker, args)

        return list(results)

    def _find_nearest_centroid(self, chunk: np.ndarray, centroids: np.ndarray) -> int:
        distances = np.linalg.norm(centroids - chunk, axis=1)
        return int(np.argmin(distances))

    def _compress_embedding(self, embedding: np.ndarray, centroids: List[np.ndarray]) -> np.ndarray:
        compressed_indices = []
        for i in range(self.chunks):
            chunk = embedding[i * self.subdim:(i + 1) * self.subdim]
            nearest_centroid_idx = self._find_nearest_centroid(chunk, centroids[i])
            compressed_indices.append(nearest_centroid_idx)
        return np.array(compressed_indices, dtype=np.int64)

    def compress(self, embeddings: np.ndarray) -> np.ndarray:
        self._validate_embeddings(embeddings)
        
        # Split embeddings into chunks
        chunks = self._chunk_embeddings(embeddings)
        
        centroids = self._compute_centroids(chunks)
        self.centroids = centroids
        
        # Compress each embedding
        compressed_embeddings = [
            self._compress_embedding(embedding, centroids)
            for embedding in embeddings
        ]
        
        return np.array(compressed_embeddings, dtype=np.int64)

    
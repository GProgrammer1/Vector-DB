import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union

from scipy.cluster.vq import kmeans2
from ..util.distance import euclidean_vector_distance
from ..types import Node
from ..storage import NodeStorage, InMemoryNodeStorage


class IvfIndex:

    def __init__(
        self,
        k: int,
        storage: Optional[NodeStorage] = None,
        index_file: Optional[Union[str, Path]] = None,
    ):
        """
        Args:
            k: Number of clusters (centroids) for k-means
            storage: NodeStorage backend for embeddings and metadata
            index_file: Optional path to save/load index state
        """
        if k <= 0:
            raise ValueError("k-means parameter should be positive")
        self.k = k
        self.storage = storage or InMemoryNodeStorage()
        self.index_file = Path(index_file) if index_file else None

        self.centroids: Optional[np.ndarray] = None
        self.inverted_lists: List[List[int]] = []

        if self.index_file and self.index_file.exists():
            self.load_index()

    def build_index(self, nodes: List[Node]) -> None:
        if not nodes:
            raise ValueError("Cannot build index with empty node list")

        for node in nodes:
            self.storage.save(node)

        embeddings = np.array([node.embedding for node in nodes])
        ids = [node.id for node in nodes]

        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2D array, got {embeddings.ndim}D")
        if embeddings.shape[0] < self.k:
            raise ValueError(f"Need at least {self.k} vectors for {self.k} clusters")

        self.centroids, labels = kmeans2(embeddings, self.k, iter=100, minit='points')

        self.inverted_lists = [[] for _ in range(self.k)]
        for i, label in enumerate(labels):
            node_id = ids[i]
            self.inverted_lists[label].append(node_id)

        if self.index_file:
            self.save_index()

    def add(self, node: Node) -> None:
        if self.centroids is None:
            raise ValueError("Index must be built before adding nodes")

        embedding = node.embedding
        if embedding.ndim != 1:
            raise ValueError("embedding must be 1D array")
        if embedding.shape[0] != self.centroids.shape[1]:
            raise ValueError(
                f"embedding dimension {embedding.shape[0]} doesn't match "
                f"centroid dimension {self.centroids.shape[1]}"
            )

        self.storage.save(node)

        distances = np.linalg.norm(self.centroids - embedding, axis=1)
        nearest_cluster = int(np.argmin(distances))

        self.inverted_lists[nearest_cluster].append(node.id)

        if self.index_file:
            self.save_index()

    def delete(self, node_id: int) -> None:
    
        for cluster_list in self.inverted_lists:
            if node_id in cluster_list:
                cluster_list.remove(node_id)

        if hasattr(self.storage, "delete"):
            self.storage.delete(node_id)

        if self.index_file:
            self.save_index()

    def search(
        self,
        query: np.ndarray,
        n_probe: int,
        top_k: int,
    ) -> List[Tuple[Node, float]]:
        """

        Args:
            query: Query vector of shape (dim,)
            n_probe: Number of clusters to probe
            top_k: Number of nearest neighbors to return

        Returns:
            List of (Node, distance) tuples, sorted by distance
        """
        if self.centroids is None:
            raise ValueError("Index must be built before searching")

        if query.ndim != 1:
            raise ValueError("query must be 1D array")
        if query.shape[0] != self.centroids.shape[1]:
            raise ValueError(
                f"query dimension {query.shape[0]} doesn't match "
                f"centroid dimension {self.centroids.shape[1]}"
            )

        if n_probe <= 0 or n_probe > self.k:
            raise ValueError(f"n_probe must be between 1 and {self.k}")

        # Find n_probe nearest centroids
        centroid_distances = np.linalg.norm(self.centroids - query, axis=1)
        nearest_clusters = np.argsort(centroid_distances)[:n_probe]

        # Search in selected clusters
        candidates: List[Tuple[float, int]] = []

        for cluster_id in nearest_clusters:
            # Get all node IDs in this cluster
            node_ids = self.inverted_lists[cluster_id]

            # Compute distances to all vectors in this cluster
            for node_id in node_ids:
                try:
                    emb = self.storage.get_embedding(node_id)
                    dist = euclidean_vector_distance(query, emb)
                    candidates.append((dist, node_id))
                except KeyError:
                    # Node was deleted from storage but not yet from inverted list
                    continue

        # Sort by distance and return top_k
        candidates.sort(key=lambda x: x[0])

        # Return (Node, distance)
        results = []
        for dist, node_id in candidates[:top_k]:
            node = self.storage.get(node_id)
            if node:
                results.append((node, dist))
        return results

    def get_cluster_size(self, cluster_id: int) -> int:
        """Get the number of vectors in a specific cluster."""
        if cluster_id < 0 or cluster_id >= self.k:
            raise ValueError(f"cluster_id must be between 0 and {self.k - 1}")
        return len(self.inverted_lists[cluster_id])

    def get_cluster_stats(self) -> dict:
        """Get statistics about cluster sizes."""
        sizes = [len(lst) for lst in self.inverted_lists]
        return {
            "min_size": min(sizes),
            "max_size": max(sizes),
            "avg_size": sum(sizes) / len(sizes) if sizes else 0,
            "total_vectors": sum(sizes),
        }

    def save_index(self) -> None:
        if self.index_file is None:
            return

        index_data = {
            "k": self.k,
            "centroids": self.centroids,
            "inverted_lists": self.inverted_lists,
        }

        with open(self.index_file, "wb") as f:
            pickle.dump(index_data, f)

    def load_index(self) -> None:
        if self.index_file is None or not self.index_file.exists():
            return

        with open(self.index_file, "rb") as f:
            index_data = pickle.load(f)

        self.k = index_data["k"]
        self.centroids = index_data["centroids"]
        self.inverted_lists = index_data["inverted_lists"]

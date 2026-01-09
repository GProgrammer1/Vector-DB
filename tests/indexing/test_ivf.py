"""Unit tests for IVF index with storage abstraction."""

import numpy as np
import pytest
import tempfile
from pathlib import Path

from vector_db.indexing.ivf import IvfIndex
from vector_db.types import Node
from vector_db.storage import InMemoryNodeStorage, MMapNodeStorage
from vector_db.util.distance import euclidean_vector_distance


class TestIVF:
    """Test suite for IVF index."""

    def test_initialization(self):
        """Test IVF initialization."""
        ivf = IvfIndex(k=10)
        
        assert ivf.k == 10
        assert ivf.centroids is None
        assert len(ivf.inverted_lists) == 0

    def test_build_index(self):
        """Test building IVF index."""
        k = 4
        dim = 8
        num_nodes = 50
        
        storage = InMemoryNodeStorage()
        ivf = IvfIndex(k=k, storage=storage)
        
        nodes = [
            Node(id=i, embedding=np.random.rand(dim).astype(np.float32), content=f"node_{i}")
            for i in range(num_nodes)
        ]
        
        ivf.build_index(nodes)
        
        assert ivf.centroids is not None
        assert ivf.centroids.shape == (k, dim)
        assert len(ivf.inverted_lists) == k
        assert sum(len(lst) for lst in ivf.inverted_lists) == num_nodes
        assert storage.size() == num_nodes

    def test_build_index_empty_list(self):
        """Test building index with empty node list."""
        ivf = IvfIndex(k=4)
        
        with pytest.raises(ValueError, match="Cannot build index with empty node list"):
            ivf.build_index([])

    def test_build_index_insufficient_nodes(self):
        """Test building index with fewer nodes than clusters."""
        ivf = IvfIndex(k=10)
        
        nodes = [
            Node(id=i, embedding=np.random.rand(8).astype(np.float32))
            for i in range(5)  # Less than k=10
        ]
        
        with pytest.raises(ValueError, match="Need at least 10 vectors for 10 clusters"):
            ivf.build_index(nodes)

    def test_add_node(self):
        """Test adding a node to existing index."""
        k = 4
        dim = 8
        storage = InMemoryNodeStorage()
        ivf = IvfIndex(k=k, storage=storage)
        
        # Build initial index
        initial_nodes = [
            Node(id=i, embedding=np.random.rand(dim).astype(np.float32))
            for i in range(20)
        ]
        ivf.build_index(initial_nodes)
        
        initial_total = sum(len(lst) for lst in ivf.inverted_lists)
        
        # Add new node
        new_node = Node(id=99, embedding=np.random.rand(dim).astype(np.float32))
        ivf.add(new_node)
        
        assert storage.size() == 21
        assert sum(len(lst) for lst in ivf.inverted_lists) == initial_total + 1
        # New node should be in one of the inverted lists
        found = False
        for lst in ivf.inverted_lists:
            if 99 in lst:
                found = True
                break
        assert found

    def test_add_node_before_build(self):
        """Test adding node before building index."""
        ivf = IvfIndex(k=4)
        node = Node(id=0, embedding=np.random.rand(8).astype(np.float32))
        
        with pytest.raises(ValueError, match="Index must be built before adding nodes"):
            ivf.add(node)

    def test_search_basic(self):
        """Test basic search functionality."""
        k = 4
        dim = 8
        storage = InMemoryNodeStorage()
        ivf = IvfIndex(k=k, storage=storage)
        
        nodes = [
            Node(id=i, embedding=np.random.rand(dim).astype(np.float32))
            for i in range(30)
        ]
        ivf.build_index(nodes)
        
        # Search for exact match
        query = nodes[0].embedding
        results = ivf.search(query, n_probe=k, top_k=5)
        
        assert len(results) > 0
        # First result should be the query node itself
        first_node, first_dist = results[0]
        assert first_node.id == 0
        assert first_dist < 1e-6

    def test_search_recall(self):
        """Test search recall against brute force."""
        k = 4
        dim = 16
        num_nodes = 100
        storage = InMemoryNodeStorage()
        ivf = IvfIndex(k=k, storage=storage)
        
        nodes = [
            Node(id=i, embedding=np.random.rand(dim).astype(np.float32))
            for i in range(num_nodes)
        ]
        ivf.build_index(nodes)
        
        # Brute force search
        def brute_force_search(query, top_k):
            distances = []
            for node in nodes:
                dist = euclidean_vector_distance(query, node.embedding)
                distances.append((dist, node.id))
            distances.sort()
            return [node_id for _, node_id in distances[:top_k]]
        
        # Test multiple queries
        num_queries = 10
        top_k = 5
        passes = 0
        
        for _ in range(num_queries):
            query = np.random.rand(dim).astype(np.float32)
            gt_ids = set(brute_force_search(query, top_k))
            
            results = ivf.search(query, n_probe=k, top_k=top_k)
            result_ids = {node.id for node, _ in results}
            
            recall = len(gt_ids.intersection(result_ids)) / top_k
            if recall >= 0.6:  # 60% recall threshold for IVF
                passes += 1
        
        # Expect at least 70% of queries to have reasonable recall
        assert passes >= num_queries * 0.7

    def test_search_invalid_n_probe(self):
        """Test search with invalid n_probe."""
        k = 4
        dim = 8
        ivf = IvfIndex(k=k)
        
        nodes = [
            Node(id=i, embedding=np.random.rand(dim).astype(np.float32))
            for i in range(20)
        ]
        ivf.build_index(nodes)
        
        query = np.random.rand(dim).astype(np.float32)
        
        with pytest.raises(ValueError, match="n_probe must be between 1 and 4"):
            ivf.search(query, n_probe=0, top_k=5)
        
        with pytest.raises(ValueError, match="n_probe must be between 1 and 4"):
            ivf.search(query, n_probe=10, top_k=5)  # > k

    def test_search_before_build(self):
        """Test searching before building index."""
        ivf = IvfIndex(k=4)
        query = np.random.rand(8).astype(np.float32)
        
        with pytest.raises(ValueError, match="Index must be built before searching"):
            ivf.search(query, n_probe=2, top_k=5)

    def test_delete_node(self):
        """Test deleting a node from index."""
        k = 4
        dim = 8
        
        with tempfile.TemporaryDirectory() as tmpdir:
            embedding_file = Path(tmpdir) / "embeddings.npy"
            metadata_file = Path(tmpdir) / "metadata.npy"
            
            storage = MMapNodeStorage(
                embedding_file=embedding_file,
                metadata_file=metadata_file,
                dim=dim,
                capacity=100
            )
            ivf = IvfIndex(k=k, storage=storage)
            
            nodes = [
                Node(id=i, embedding=np.random.rand(dim).astype(np.float32))
                for i in range(20)
            ]
            ivf.build_index(nodes)
            
            initial_total = sum(len(lst) for lst in ivf.inverted_lists)
            
            # Delete a node
            ivf.delete(5)
            
            assert storage.get(5) is None
            assert sum(len(lst) for lst in ivf.inverted_lists) == initial_total - 1
            # Verify node is not in any inverted list
            for lst in ivf.inverted_lists:
                assert 5 not in lst

    def test_get_cluster_size(self):
        """Test getting cluster size."""
        k = 4
        dim = 8
        ivf = IvfIndex(k=k)
        
        nodes = [
            Node(id=i, embedding=np.random.rand(dim).astype(np.float32))
            for i in range(20)
        ]
        ivf.build_index(nodes)
        
        for cluster_id in range(k):
            size = ivf.get_cluster_size(cluster_id)
            assert size >= 0
            assert size == len(ivf.inverted_lists[cluster_id])
        
        with pytest.raises(ValueError):
            ivf.get_cluster_size(-1)
        
        with pytest.raises(ValueError):
            ivf.get_cluster_size(k)

    def test_get_cluster_stats(self):
        """Test getting cluster statistics."""
        k = 4
        dim = 8
        ivf = IvfIndex(k=k)
        
        nodes = [
            Node(id=i, embedding=np.random.rand(dim).astype(np.float32))
            for i in range(20)
        ]
        ivf.build_index(nodes)
        
        stats = ivf.get_cluster_stats()
        
        assert "min_size" in stats
        assert "max_size" in stats
        assert "avg_size" in stats
        assert "total_vectors" in stats
        assert stats["total_vectors"] == 20
        assert stats["min_size"] >= 0
        assert stats["max_size"] >= stats["min_size"]
        assert stats["avg_size"] == stats["total_vectors"] / k

    def test_index_persistence(self):
        """Test saving and loading index."""
        k = 4
        dim = 8
        storage = InMemoryNodeStorage()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            index_file = Path(tmpdir) / "index.pkl"
            
            # Create and build index
            ivf1 = IvfIndex(k=k, storage=storage, index_file=index_file)
            nodes = [
                Node(id=i, embedding=np.random.rand(dim).astype(np.float32))
                for i in range(20)
            ]
            ivf1.build_index(nodes)
            
            # Verify index file was created
            assert index_file.exists()
            
            # Load index
            storage2 = InMemoryNodeStorage()
            for node in nodes:
                storage2.save(node)
            
            ivf2 = IvfIndex(k=k, storage=storage2, index_file=index_file)
            
            # Verify centroids and inverted lists were loaded
            assert ivf2.centroids is not None
            np.testing.assert_array_equal(ivf2.centroids, ivf1.centroids)
            assert len(ivf2.inverted_lists) == k
            assert len(ivf2.inverted_lists) == len(ivf1.inverted_lists)
            
            # Verify search works
            query = nodes[0].embedding
            results = ivf2.search(query, n_probe=k, top_k=5)
            assert len(results) > 0

    def test_ivf_with_mmap_storage(self):
        """Test IVF with MMapNodeStorage."""
        k = 4
        dim = 8
        
        with tempfile.TemporaryDirectory() as tmpdir:
            embedding_file = Path(tmpdir) / "embeddings.npy"
            metadata_file = Path(tmpdir) / "metadata.npy"
            index_file = Path(tmpdir) / "index.pkl"
            
            storage = MMapNodeStorage(
                embedding_file=embedding_file,
                metadata_file=metadata_file,
                dim=dim,
                capacity=100
            )
            
            ivf = IvfIndex(k=k, storage=storage, index_file=index_file)
            
            # Build index
            nodes = [
                Node(id=i, embedding=np.random.rand(dim).astype(np.float32), content=f"node_{i}")
                for i in range(20)
            ]
            ivf.build_index(nodes)
            
            # Verify storage
            assert storage.size() == 20
            
            # Test search
            query = nodes[0].embedding
            results = ivf.search(query, n_probe=k, top_k=5)
            assert len(results) > 0
            
            # Test persistence
            storage.close()
            del ivf
            
            # Reload
            storage2 = MMapNodeStorage(
                embedding_file=embedding_file,
                metadata_file=metadata_file,
                dim=dim,
                capacity=100
            )
            ivf2 = IvfIndex(k=k, storage=storage2, index_file=index_file)
            
            # Verify search still works
            results2 = ivf2.search(query, n_probe=k, top_k=5)
            assert len(results2) > 0

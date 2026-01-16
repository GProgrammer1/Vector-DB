"""Unit tests for HNSW index with storage abstraction."""

import numpy as np
import pytest
import random
import tempfile
import os
from pathlib import Path

from vector_db.indexing.hnsw import HNSW
from vector_db.types import Node
from vector_db.storage import InMemoryNodeStorage, MMapNodeStorage
from vector_db.util.distance import euclidean_vector_distance


class TestHNSW:

    def test_initialization(self):
        rng = random.Random(42)
        storage = InMemoryNodeStorage()
        hnsw = HNSW(M=16, ef_construction=100, rng=rng, storage=storage)
        
        assert hnsw.M == 16
        assert hnsw.ef_construction == 100
        assert len(hnsw.node_store) == 0
        assert hnsw.entry_node_id is None
        assert hnsw.max_level == -1

    def test_insert_single_node(self):
        rng = random.Random(42)
        storage = InMemoryNodeStorage()
        hnsw = HNSW(M=16, ef_construction=100, rng=rng, storage=storage)
        
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        node = Node(id=0, embedding=vec, content="test", metadata={"key": "value"})
        hnsw.insert_node(node)
        
        assert 0 in hnsw.node_store
        assert hnsw.entry_node_id == 0
        assert hnsw.max_level >= 0
        assert hnsw.node_store[0].id == 0
        # Verify node is in storage
        stored_node = storage.get(0)
        assert stored_node is not None
        assert stored_node.id == 0
        np.testing.assert_array_equal(stored_node.embedding, vec)

    def test_insert_multiple_nodes(self):
        rng = random.Random(42)
        storage = InMemoryNodeStorage()
        hnsw = HNSW(M=4, ef_construction=10, rng=rng, storage=storage)
        
        dim = 8
        num_nodes = 20
        nodes = []
        for i in range(num_nodes):
            vec = np.random.rand(dim).astype(np.float32)
            node = Node(id=i, embedding=vec, content=f"node_{i}")
            nodes.append(node)
            hnsw.insert_node(node)
        
        assert len(hnsw.node_store) == num_nodes
        assert hnsw.entry_node_id is not None
        assert storage.size() == num_nodes

    def test_build_index(self):
        rng = random.Random(42)
        storage = InMemoryNodeStorage()
        hnsw = HNSW(M=4, ef_construction=10, rng=rng, storage=storage)
        
        dim = 4
        num_nodes = 30
        nodes = [
            Node(id=i, embedding=np.random.rand(dim).astype(np.float32), content=f"node_{i}")
            for i in range(num_nodes)
        ]
        
        hnsw.build_index(nodes)
        
        assert len(hnsw.node_store) == num_nodes
        assert hnsw.entry_node_id is not None
        assert storage.size() == num_nodes

    def test_search_basic(self):
        rng = random.Random(42)
        storage = InMemoryNodeStorage()
        hnsw = HNSW(M=4, ef_construction=10, rng=rng, storage=storage)
        
        dim = 8
        nodes = [
            Node(id=i, embedding=np.random.rand(dim).astype(np.float32))
            for i in range(20)
        ]
        hnsw.build_index(nodes)
        
        query = nodes[0].embedding
        results = hnsw.search(query, k=5, ef=20)
        
        assert len(results) > 0
        assert len(results) <= 5
        # First result should be the query node itself
        first_node, first_dist = results[0]
        assert first_node.id == 0
        assert first_dist < 1e-6

    def test_search_recall(self):
        """Test search recall against brute force."""
        rng = random.Random(42)
        storage = InMemoryNodeStorage()
        hnsw = HNSW(M=8, ef_construction=50, rng=rng, storage=storage)
        
        dim = 16
        num_nodes = 100
        nodes = [
            Node(id=i, embedding=np.random.rand(dim).astype(np.float32))
            for i in range(num_nodes)
        ]
        hnsw.build_index(nodes)
        
        def brute_force_search(query, k):
            distances = []
            for node in nodes:
                dist = euclidean_vector_distance(query, node.embedding)
                distances.append((dist, node.id))
            distances.sort()
            return [node_id for _, node_id in distances[:k]]
        
        num_queries = 10
        k = 5
        passes = 0
        
        for _ in range(num_queries):
            query = np.random.rand(dim).astype(np.float32)
            gt_ids = set(brute_force_search(query, k))
            
            results = hnsw.search(query, k=k, ef=50)
            result_ids = {node.id for node, _ in results}
            
            recall = len(gt_ids.intersection(result_ids)) / k
            if recall >= 0.7: 
                passes += 1
        
        assert passes >= num_queries * 0.8

    def test_graph_connectivity(self):
        rng = random.Random(42)
        storage = InMemoryNodeStorage()
        hnsw = HNSW(M=4, ef_construction=20, rng=rng, storage=storage)
        
        dim = 8
        num_nodes = 50
        nodes = [
            Node(id=i, embedding=np.random.rand(dim).astype(np.float32))
            for i in range(num_nodes)
        ]
        hnsw.build_index(nodes)
        
        # BFS from entry node at level 0
        visited = set()
        queue = [hnsw.entry_node_id]
        visited.add(hnsw.entry_node_id)
        
        while queue:
            curr_id = queue.pop(0)
            if curr_id not in hnsw.node_store:
                continue
            curr_node = hnsw.node_store[curr_id]
            for neighbor_id in curr_node.neighbors.get(0, []):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append(neighbor_id)
        
        assert len(visited) >= num_nodes * 0.9

    def test_delete_node(self):
        rng = random.Random(42)
        # Use MMapNodeStorage which supports delete
        with tempfile.TemporaryDirectory() as tmpdir:
            embedding_file = Path(tmpdir) / "embeddings.npy"
            metadata_file = Path(tmpdir) / "metadata.npy"
            
            storage = MMapNodeStorage(
                embedding_file=embedding_file,
                metadata_file=metadata_file,
                dim=8,
                capacity=100
            )
            hnsw = HNSW(M=4, ef_construction=10, rng=rng, storage=storage)
            
            dim = 8
            nodes = [
                Node(id=i, embedding=np.random.rand(dim).astype(np.float32))
                for i in range(20)
            ]
            hnsw.build_index(nodes)
            
            initial_size = len(hnsw.node_store)
            assert initial_size == 20
            
            hnsw.delete_node(5)
            
            assert 5 not in hnsw.node_store
            assert storage.get(5) is None
            assert len(hnsw.node_store) == initial_size - 1

    def test_index_persistence(self):
        rng = random.Random(42)
        storage = InMemoryNodeStorage()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            index_file = Path(tmpdir) / "index.pkl"
            
            # Create and build index
            hnsw1 = HNSW(M=4, ef_construction=10, rng=rng, storage=storage, index_file=index_file)
            dim = 8
            nodes = [
                Node(id=i, embedding=np.random.rand(dim).astype(np.float32))
                for i in range(20)
            ]
            hnsw1.build_index(nodes)
            
            assert index_file.exists()
            
            # Load index
            storage2 = InMemoryNodeStorage()
            for node in nodes:
                storage2.save(node)
            
            hnsw2 = HNSW(M=4, ef_construction=10, rng=rng, storage=storage2, index_file=index_file)
            
            # Verify graph structure was loaded
            assert len(hnsw2.node_store) == 20
            assert hnsw2.entry_node_id == hnsw1.entry_node_id
            assert hnsw2.max_level == hnsw1.max_level
            
            query = nodes[0].embedding
            results = hnsw2.search(query, k=5, ef=20)
            assert len(results) > 0

    def test_hnsw_with_mmap_storage(self):
        rng = random.Random(42)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            embedding_file = Path(tmpdir) / "embeddings.npy"
            metadata_file = Path(tmpdir) / "metadata.npy"
            index_file = Path(tmpdir) / "index.pkl"
            
            storage = MMapNodeStorage(
                embedding_file=embedding_file,
                metadata_file=metadata_file,
                dim=8,
                capacity=100
            )
            
            hnsw = HNSW(M=4, ef_construction=10, rng=rng, storage=storage, index_file=index_file)
            
            dim = 8
            nodes = [
                Node(id=i, embedding=np.random.rand(dim).astype(np.float32), content=f"node_{i}")
                for i in range(20)
            ]
            hnsw.build_index(nodes)
            
            assert storage.size() == 20
            
            query = nodes[0].embedding
            results = hnsw.search(query, k=5, ef=20)
            assert len(results) > 0
            
            storage.close()
            del hnsw
            
            # Reload
            storage2 = MMapNodeStorage(
                embedding_file=embedding_file,
                metadata_file=metadata_file,
                dim=8,
                capacity=100
            )
            hnsw2 = HNSW(M=4, ef_construction=10, rng=rng, storage=storage2, index_file=index_file)
            
            results2 = hnsw2.search(query, k=5, ef=20)
            assert len(results2) > 0

    def test_idempotent_insert(self):
        rng = random.Random(42)
        storage = InMemoryNodeStorage()
        hnsw = HNSW(M=4, ef_construction=10, rng=rng, storage=storage)
        
        vec = np.random.rand(8).astype(np.float32)
        node = Node(id=0, embedding=vec)
        
        # Insert twice
        hnsw.insert_node(node)
        initial_neighbors = hnsw.node_store[0].neighbors.copy()
        
        hnsw.insert_node(node)
        
        assert len(hnsw.node_store) == 1
        assert hnsw.node_store[0].neighbors == initial_neighbors

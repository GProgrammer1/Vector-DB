
import numpy as np
import pytest
import tempfile
import os
import yaml
from pathlib import Path

from vector_db.inference.mmap_vector_store import MemoryMappingService


class TestMemoryMappingService:

    def _create_temp_config(self, tmpdir: str) -> str:
        config_path = os.path.join(tmpdir, "config.yaml")
        config = {
            "index": {
                "M": 4,
                "ef_construction": 10
            }
        }
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        return config_path

    def test_init_valid_params(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._create_temp_config(tmpdir)
            tmp_path = os.path.join(tmpdir, "vectors")
            
            service = MemoryMappingService(
                file_path=tmp_path,
                dim=128,
                capacity=1000,
                config_path=config_path
            )
            assert service.dim == 128
            assert service.capacity == 1000
            assert service.size == 0
            assert service.file_path == Path(tmp_path)

    def test_init_invalid_dim(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._create_temp_config(tmpdir)
            tmp_path = os.path.join(tmpdir, "vectors")
            
            with pytest.raises(ValueError, match="Dimension must be greater than 0"):
                MemoryMappingService(
                    file_path=tmp_path,
                    dim=0,
                    capacity=1000,
                    config_path=config_path
                )

    def test_init_invalid_capacity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._create_temp_config(tmpdir)
            tmp_path = os.path.join(tmpdir, "vectors")
            
            with pytest.raises(ValueError, match="Capacity must be greater than 0"):
                MemoryMappingService(
                    file_path=tmp_path,
                    dim=128,
                    capacity=0,
                    config_path=config_path
                )

    def test_init_missing_config_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = os.path.join(tmpdir, "vectors")
            
            with pytest.raises(ValueError, match="config_path is required"):
                MemoryMappingService(
                    file_path=tmp_path,
                    dim=128,
                    capacity=1000
                )

    def test_write_single_embedding(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._create_temp_config(tmpdir)
            tmp_path = os.path.join(tmpdir, "vectors")
            
            service = MemoryMappingService(
                file_path=tmp_path,
                dim=4,
                capacity=10,
                config_path=config_path
            )
            embedding = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
            
            node_id = service.write(embedding)
            
            assert isinstance(node_id, int)
            assert node_id >= 0
            assert service.size == 1

    def test_write_multiple_embeddings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._create_temp_config(tmpdir)
            tmp_path = os.path.join(tmpdir, "vectors")
            
            service = MemoryMappingService(
                file_path=tmp_path,
                dim=4,
                capacity=10,
                config_path=config_path
            )
            embeddings = [
                np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32),
                np.array([9.0, 10.0, 11.0, 12.0], dtype=np.float32),
            ]
            
            node_ids = [service.write(emb) for emb in embeddings]
            
            assert len(node_ids) == 3
            assert all(isinstance(nid, int) for nid in node_ids)
            assert service.size == 3

    def test_write_with_content_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._create_temp_config(tmpdir)
            tmp_path = os.path.join(tmpdir, "vectors")
            
            service = MemoryMappingService(
                file_path=tmp_path,
                dim=4,
                capacity=10,
                config_path=config_path
            )
            embedding = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
            
            node_id = service.write(
                embedding,
                content="test content",
                metadata={"key": "value", "num": 42}
            )
            
            node = service.read(node_id)
            assert node.content == "test content"
            assert node.metadata["key"] == "value"
            assert node.metadata["num"] == 42

    def test_write_invalid_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._create_temp_config(tmpdir)
            tmp_path = os.path.join(tmpdir, "vectors")
            
            service = MemoryMappingService(
                file_path=tmp_path,
                dim=4,
                capacity=10,
                config_path=config_path
            )
            with pytest.raises(TypeError, match="Embedding must be a numpy array"):
                service.write([1.0, 2.0, 3.0, 4.0])

    def test_write_wrong_dimension(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._create_temp_config(tmpdir)
            tmp_path = os.path.join(tmpdir, "vectors")
            
            service = MemoryMappingService(
                file_path=tmp_path,
                dim=4,
                capacity=10,
                config_path=config_path
            )
            embedding = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
            with pytest.raises(ValueError, match="Embedding must be a 1D array"):
                service.write(embedding)

    def test_write_wrong_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._create_temp_config(tmpdir)
            tmp_path = os.path.join(tmpdir, "vectors")
            
            service = MemoryMappingService(
                file_path=tmp_path,
                dim=4,
                capacity=10,
                config_path=config_path
            )
            embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)  # size 3 instead of 4
            with pytest.raises(ValueError, match="Embedding must be of dimension 4"):
                service.write(embedding)

    def test_read_single_embedding(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._create_temp_config(tmpdir)
            tmp_path = os.path.join(tmpdir, "vectors")
            
            service = MemoryMappingService(
                file_path=tmp_path,
                dim=4,
                capacity=10,
                config_path=config_path
            )
            embedding = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
            node_id = service.write(embedding)
            
            node = service.read(node_id)
            
            assert node.id == node_id
            np.testing.assert_array_equal(node.embedding, embedding)

    def test_read_multiple_embeddings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._create_temp_config(tmpdir)
            tmp_path = os.path.join(tmpdir, "vectors")
            
            service = MemoryMappingService(
                file_path=tmp_path,
                dim=4,
                capacity=10,
                config_path=config_path
            )
            embeddings = [
                np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32),
                np.array([9.0, 10.0, 11.0, 12.0], dtype=np.float32),
            ]
            
            node_ids = [service.write(emb) for emb in embeddings]
            
            # Read all embeddings
            for node_id, expected_emb in zip(node_ids, embeddings):
                node = service.read(node_id)
                np.testing.assert_array_equal(node.embedding, expected_emb)

    def test_read_invalid_index_type(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._create_temp_config(tmpdir)
            tmp_path = os.path.join(tmpdir, "vectors")
            
            service = MemoryMappingService(
                file_path=tmp_path,
                dim=4,
                capacity=10,
                config_path=config_path
            )
            with pytest.raises(TypeError, match="Node ID must be an integer"):
                service.read("0")

    def test_read_nonexistent_node(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._create_temp_config(tmpdir)
            tmp_path = os.path.join(tmpdir, "vectors")
            
            service = MemoryMappingService(
                file_path=tmp_path,
                dim=4,
                capacity=10,
                config_path=config_path
            )
            
            with pytest.raises(IndexError, match="Node 999 not found"):
                service.read(999)

    def test_get_embedding(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._create_temp_config(tmpdir)
            tmp_path = os.path.join(tmpdir, "vectors")
            
            service = MemoryMappingService(
                file_path=tmp_path,
                dim=4,
                capacity=10,
                config_path=config_path
            )
            embedding = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
            node_id = service.write(embedding)
            
            retrieved_emb = service.get_embedding(node_id)
            np.testing.assert_array_equal(retrieved_emb, embedding)

    def test_delete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._create_temp_config(tmpdir)
            tmp_path = os.path.join(tmpdir, "vectors")
            
            service = MemoryMappingService(
                file_path=tmp_path,
                dim=4,
                capacity=10,
                config_path=config_path
            )
            embedding = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
            node_id = service.write(embedding)
            
            assert service.size == 1
            
            service.delete(node_id)
            
            assert service.size == 0
            with pytest.raises(IndexError):
                service.read(node_id)

    def test_search(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._create_temp_config(tmpdir)
            tmp_path = os.path.join(tmpdir, "vectors")
            
            service = MemoryMappingService(
                file_path=tmp_path,
                dim=4,
                capacity=10,
                config_path=config_path
            )
            # Insert some embeddings
            embeddings = [
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
                np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
            ]
            for emb in embeddings:
                service.write(emb)
            
            # Search for nearest neighbor
            query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            results = service.search(query, k=2, ef=10)
            
            assert len(results) > 0
            assert len(results) <= 2
            
            first_node, first_dist = results[0]
            assert first_dist < 1e-6

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = self._create_temp_config(tmpdir)
            tmp_path = os.path.join(tmpdir, "vectors")
            index_file = os.path.join(tmpdir, "index.pkl")
            
            # Write data
            service1 = MemoryMappingService(
                file_path=tmp_path,
                dim=4,
                capacity=10,
                config_path=config_path,
                index_file=index_file
            )
            embeddings = [
                np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32),
            ]
            node_ids = [service1.write(emb, content=f"node_{i}") for i, emb in enumerate(embeddings)]
            del service1  # Close the file
            
            # Reopen and read
            service2 = MemoryMappingService(
                file_path=tmp_path,
                dim=4,
                capacity=10,
                config_path=config_path,
                index_file=index_file
            )
            # Verify data persisted
            assert service2.size == 2
            for node_id, expected_emb in zip(node_ids, embeddings):
                node = service2.read(node_id)
                np.testing.assert_array_equal(node.embedding, expected_emb)

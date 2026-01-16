"""Memory-mapped file-based IPC for inter-process communication."""

import struct
import mmap
import json
import time
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, List
from enum import IntEnum
import numpy as np


class MessageType(IntEnum):
    """Message types for IPC communication."""
    EMBED_REQUEST = 1
    EMBED_RESPONSE = 2
    BATCH_EMBED_REQUEST = 3
    BATCH_EMBED_RESPONSE = 4
    HEALTH_CHECK = 5
    HEALTH_RESPONSE = 6


class MMapIPC:
    """
    Memory-mapped file-based IPC for communication between services.
    
    Uses shared memory files that can be mounted as volumes in containers.
    """
    
    # Header structure: message_type (4 bytes) + message_id (36 bytes UUID) + 
    # data_length (8 bytes) + status (4 bytes) = 52 bytes
    HEADER_SIZE = 52
    MAX_MESSAGE_SIZE = 50 * 1024 * 1024  # 50MB max message size (matches document limit)
    QUEUE_SIZE = 10  # Maximum number of pending messages (small queue to avoid huge files)
    
    def __init__(self, shared_dir: str, role: str = "client"):
        """
        
        Args:
            shared_dir: Directory for shared memory files (should be mounted volume in containers)
            role: "client" or "server" - determines which files to use
        """
        self.shared_dir = Path(shared_dir)
        self.shared_dir.mkdir(parents=True, exist_ok=True)
        self.role = role
        
        if role == "client":
            # Client writes requests, reads responses
            self.request_file = self.shared_dir / "embedding_requests.mmap"
            self.response_file = self.shared_dir / "embedding_responses.mmap"
        else:
            # Server reads requests, writes responses
            self.request_file = self.shared_dir / "embedding_requests.mmap"
            self.response_file = self.shared_dir / "embedding_responses.mmap"
        
        # Initialize mmap files if they don't exist
        self._init_mmap_files()
    
    def _init_mmap_files(self):
        """Initialize memory-mapped files."""
        # Calculate file size: header + (max message size * queue size) per slot
        slot_size = self.HEADER_SIZE + self.MAX_MESSAGE_SIZE
        file_size = slot_size * self.QUEUE_SIZE
        
        for mmap_file in [self.request_file, self.response_file]:
            if not mmap_file.exists():
                # Create file with zeros using sparse file creation
                # Write in chunks to avoid memory issues
                chunk_size = 1024 * 1024  # 1MB chunks
                with open(mmap_file, 'wb') as f:
                    remaining = file_size
                    while remaining > 0:
                        write_size = min(chunk_size, remaining)
                        f.write(b'\x00' * write_size)
                        remaining -= write_size
    
    def _pack_header(self, msg_type: MessageType, msg_id: str, data_length: int, status: int = 0) -> bytes:
        """Pack message header."""
        msg_id_bytes = msg_id.encode('utf-8')[:36].ljust(36, b'\x00')
        return struct.pack('>I36sQI', int(msg_type), msg_id_bytes, data_length, status)
    
    def _unpack_header(self, data: bytes) -> tuple:
        """Unpack message header."""
        msg_type, msg_id_bytes, data_length, status = struct.unpack('>I36sQI', data[:self.HEADER_SIZE])
        msg_id = msg_id_bytes.rstrip(b'\x00').decode('utf-8')
        return MessageType(msg_type), msg_id, data_length, status
    
    def _write_message(self, mmap_file: Path, slot: int, msg_type: MessageType, 
                      data: bytes, msg_id: Optional[str] = None) -> str:
        """Write a message to a specific slot in the mmap file."""
        if msg_id is None:
            msg_id = str(uuid.uuid4())
        
        with open(mmap_file, 'r+b') as f:
            with mmap.mmap(f.fileno(), 0) as mm:
                offset = slot * (self.HEADER_SIZE + self.MAX_MESSAGE_SIZE)
                
                # Write header
                header = self._pack_header(msg_type, msg_id, len(data))
                mm[offset:offset + self.HEADER_SIZE] = header
                
                # Write data
                data_offset = offset + self.HEADER_SIZE
                mm[data_offset:data_offset + len(data)] = data
        
        return msg_id
    
    def _read_message(self, mmap_file: Path, slot: int) -> Optional[tuple]:
        """Read a message from a specific slot."""
        with open(mmap_file, 'r+b') as f:
            with mmap.mmap(f.fileno(), 0) as mm:
                offset = slot * (self.HEADER_SIZE + self.MAX_MESSAGE_SIZE)
                
                # Read header
                header_data = mm[offset:offset + self.HEADER_SIZE]
                msg_type, msg_id, data_length, status = self._unpack_header(header_data)
                
                if data_length == 0:
                    return None  # Empty slot
                
                # Read data
                data_offset = offset + self.HEADER_SIZE
                data = mm[data_offset:data_offset + data_length]
                
                return msg_type, msg_id, data, status
    
    def _find_empty_slot(self, mmap_file: Path) -> Optional[int]:
        """Find an empty slot in the mmap file."""
        with open(mmap_file, 'r+b') as f:
            with mmap.mmap(f.fileno(), 0) as mm:
                for i in range(self.QUEUE_SIZE):
                    offset = i * (self.HEADER_SIZE + self.MAX_MESSAGE_SIZE)
                    header_data = mm[offset:offset + self.HEADER_SIZE]
                    _, _, data_length, _ = self._unpack_header(header_data)
                    if data_length == 0:
                        return i
        return None
    
    def _find_message_by_id(self, mmap_file: Path, msg_id: str) -> Optional[int]:
        """Find a message slot by message ID."""
        with open(mmap_file, 'r+b') as f:
            with mmap.mmap(f.fileno(), 0) as mm:
                for i in range(self.QUEUE_SIZE):
                    offset = i * (self.HEADER_SIZE + self.MAX_MESSAGE_SIZE)
                    header_data = mm[offset:offset + self.HEADER_SIZE]
                    _, slot_msg_id, data_length, _ = self._unpack_header(header_data)
                    if slot_msg_id == msg_id and data_length > 0:
                        return i
        return None
    
    def _clear_slot(self, mmap_file: Path, slot: int):
        """Clear a slot in the mmap file."""
        with open(mmap_file, 'r+b') as f:
            with mmap.mmap(f.fileno(), 0) as mm:
                offset = slot * (self.HEADER_SIZE + self.MAX_MESSAGE_SIZE)
                # Write zero header to clear
                mm[offset:offset + self.HEADER_SIZE] = b'\x00' * self.HEADER_SIZE
    
    def send_embed_request(self, text: str, timeout: float = 30.0) -> np.ndarray:
       
        # Find empty slot
        slot = self._find_empty_slot(self.request_file)
        if slot is None:
            raise RuntimeError("No available slots in request queue")
        
        # Prepare request data
        request_data = json.dumps({"text": text}).encode('utf-8')
        msg_id = self._write_message(
            self.request_file, slot, MessageType.EMBED_REQUEST, request_data
        )
        
        # Wait for response
        start_time = time.time()
        while time.time() - start_time < timeout:
            response_slot = self._find_message_by_id(self.response_file, msg_id)
            if response_slot is not None:
                result = self._read_message(self.response_file, response_slot)
                if result:
                    msg_type, resp_msg_id, data, status = result
                    if resp_msg_id == msg_id and msg_type == MessageType.EMBED_RESPONSE:
                        # Clear slots
                        self._clear_slot(self.request_file, slot)
                        self._clear_slot(self.response_file, response_slot)
                        
                        if status != 0:
                            error_msg = data.decode('utf-8')
                            raise RuntimeError(f"Embedding failed: {error_msg}")
                        
                        # Parse response
                        response_data = json.loads(data.decode('utf-8'))
                        return np.array(response_data["embedding"], dtype=np.float32)
            
            time.sleep(0.01)  # Small delay to avoid busy waiting
        
        # Cleanup on timeout
        self._clear_slot(self.request_file, slot)
        raise TimeoutError(f"Embedding request timed out after {timeout}s")
    
    def get_pending_requests(self) -> List[tuple]:
        """Get all pending requests (server side)."""
        requests = []
        with open(self.request_file, 'r+b') as f:
            with mmap.mmap(f.fileno(), 0) as mm:
                for i in range(self.QUEUE_SIZE):
                    offset = i * (self.HEADER_SIZE + self.MAX_MESSAGE_SIZE)
                    header_data = mm[offset:offset + self.HEADER_SIZE]
                    msg_type, msg_id, data_length, _ = self._unpack_header(header_data)
                    
                    if data_length > 0 and msg_type in (MessageType.EMBED_REQUEST, MessageType.BATCH_EMBED_REQUEST):
                        data_offset = offset + self.HEADER_SIZE
                        data = mm[data_offset:data_offset + data_length]
                        requests.append((i, msg_type, msg_id, data))
        
        return requests
    
    def send_embed_response(self, msg_id: str, embedding: Optional[np.ndarray] = None, error: Optional[str] = None):
        """Send embedding response (server side)."""
        slot = self._find_empty_slot(self.response_file)
        if slot is None:
            raise RuntimeError("No available slots in response queue")
        
        if error:
            status = 1
            data = error.encode('utf-8')
        else:
            if embedding is None:
                raise ValueError("Either embedding or error must be provided")
            status = 0
            response_data = {
                "embedding": embedding.tolist(),
                "dimension": embedding.shape[0]
            }
            data = json.dumps(response_data).encode('utf-8')
        
        self._write_message(
            self.response_file, slot, MessageType.EMBED_RESPONSE, data, msg_id
        )
    
    def send_batch_embed_request(self, texts: List[str], timeout: float = 30.0) -> np.ndarray:
        """Send a batch embedding request and wait for response."""
        # Find empty slot
        slot = self._find_empty_slot(self.request_file)
        if slot is None:
            raise RuntimeError("No available slots in request queue")
        
        # Prepare request data
        request_data = json.dumps({"texts": texts}).encode('utf-8')
        msg_id = self._write_message(
            self.request_file, slot, MessageType.BATCH_EMBED_REQUEST, request_data
        )
        
        # Wait for response
        start_time = time.time()
        while time.time() - start_time < timeout:
            response_slot = self._find_message_by_id(self.response_file, msg_id)
            if response_slot is not None:
                result = self._read_message(self.response_file, response_slot)
                if result:
                    msg_type, resp_msg_id, data, status = result
                    if resp_msg_id == msg_id and msg_type == MessageType.BATCH_EMBED_RESPONSE:
                        # Clear slots
                        self._clear_slot(self.request_file, slot)
                        self._clear_slot(self.response_file, response_slot)
                        
                        if status != 0:
                            error_msg = data.decode('utf-8')
                            raise RuntimeError(f"Batch embedding failed: {error_msg}")
                        
                        # Parse response
                        response_data = json.loads(data.decode('utf-8'))
                        embeddings_list = response_data["embeddings"]
                        return np.array(embeddings_list, dtype=np.float32)
            
            time.sleep(0.01)  # Small delay to avoid busy waiting
        
        # Cleanup on timeout
        self._clear_slot(self.request_file, slot)
        raise TimeoutError(f"Batch embedding request timed out after {timeout}s")
    
    def send_batch_embed_response(self, msg_id: str, embeddings: Optional[np.ndarray] = None, error: Optional[str] = None):
        """Send batch embedding response (server side)."""
        slot = self._find_empty_slot(self.response_file)
        if slot is None:
            raise RuntimeError("No available slots in response queue")
        
        if error:
            status = 1
            data = error.encode('utf-8')
        else:
            if embeddings is None:
                raise ValueError("Either embeddings or error must be provided")
            status = 0
            response_data = {
                "embeddings": [emb.tolist() for emb in embeddings],
                "dimension": embeddings.shape[1] if len(embeddings.shape) > 1 else embeddings.shape[0],
                "count": embeddings.shape[0]
            }
            data = json.dumps(response_data).encode('utf-8')
        
        self._write_message(
            self.response_file, slot, MessageType.BATCH_EMBED_RESPONSE, data, msg_id
        )


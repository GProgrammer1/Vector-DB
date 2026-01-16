"""Mmap-based IPC client for embedding service."""

import numpy as np
import os
from typing import List, Optional
from pathlib import Path

from .mmap_ipc import MMapIPC


class EmbeddingClient:
    def __init__(
        self,
        timeout: float = 30.0,
        use_async: bool = False,
        mmap_shared_dir: Optional[str] = None,
    ):
       
        self.timeout = timeout
        self.use_async = use_async
        
        if mmap_shared_dir is None:
            mmap_shared_dir = os.getenv("MMAP_SHARED_DIR", "/tmp/vector_db_shared")
        
        self.mmap_ipc = MMapIPC(mmap_shared_dir, role="client")

    def embed_text(self, text: str) -> np.ndarray:
        if self.use_async:
            raise RuntimeError("Use embed_text_async() for async client")
        return self.mmap_ipc.send_embed_request(text, timeout=self.timeout)

    async def embed_text_async(self, text: str) -> np.ndarray:
        if not self.use_async:
            raise RuntimeError("Use embed_text() for sync client")
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.mmap_ipc.send_embed_request, text, self.timeout
        )

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if self.use_async:
            raise RuntimeError("Use embed_texts_async() for async client")
        return self.mmap_ipc.send_batch_embed_request(texts, timeout=self.timeout)

    async def embed_texts_async(self, texts: List[str]) -> np.ndarray:
        if not self.use_async:
            raise RuntimeError("Use embed_texts() for sync client")
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.mmap_ipc.send_batch_embed_request, texts, self.timeout
        )

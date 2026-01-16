"""HTTP client for embedding service API using httpx."""

import numpy as np
import time
from typing import List, Optional
import httpx
from pathlib import Path


class EmbeddingClient:
    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        max_retries: int = 5,
        retry_delay: float = 1.0,
        use_async: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.use_async = use_async
        
        if use_async:
            self._client = httpx.AsyncClient(timeout=timeout)
        else:
            self._client = httpx.Client(timeout=timeout)
    
    def _retry_request(self, method, *args, **kwargs):
        last_exception = None
        delay = self.retry_delay
        
        for attempt in range(self.max_retries):
            try:
                if method == "get":
                    return self._client.get(*args, **kwargs)
                elif method == "post":
                    return self._client.post(*args, **kwargs)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
            except (httpx.ConnectError, httpx.NetworkError, httpx.TimeoutException) as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise
            except httpx.HTTPStatusError:
                raise
        
        if last_exception:
            raise last_exception

    async def _retry_request_async(self, method, *args, **kwargs):
        import asyncio
        last_exception = None
        delay = self.retry_delay
        
        for attempt in range(self.max_retries):
            try:
                if method == "get":
                    return await self._client.get(*args, **kwargs)
                elif method == "post":
                    return await self._client.post(*args, **kwargs)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
            except (httpx.ConnectError, httpx.NetworkError, httpx.TimeoutException) as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    raise
            except httpx.HTTPStatusError:
                raise
        
        if last_exception:
            raise last_exception

    def embed_text(self, text: str) -> np.ndarray:
        if self.use_async:
            raise RuntimeError("Use embed_text_async() for async client")
        response = self._retry_request(
            "post",
            f"{self.base_url}/embed",
            json={"text": text}
        )
        response.raise_for_status()
        data = response.json()
        return np.array(data["embedding"], dtype=np.float32)

    async def embed_text_async(self, text: str) -> np.ndarray:
        if not self.use_async:
            raise RuntimeError("Use embed_text() for sync client")
        response = await self._retry_request_async(
            "post",
            f"{self.base_url}/embed",
            json={"text": text}
        )
        response.raise_for_status()
        data = response.json()
        return np.array(data["embedding"], dtype=np.float32)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if self.use_async:
            raise RuntimeError("Use embed_texts_async() for async client")
        response = self._retry_request(
            "post",
            f"{self.base_url}/embed/batch",
            json={"texts": texts}
        )
        response.raise_for_status()
        data = response.json()
        embeddings = np.array(data["embeddings"], dtype=np.float32)
        return embeddings

    async def embed_texts_async(self, texts: List[str]) -> np.ndarray:
        if not self.use_async:
            raise RuntimeError("Use embed_texts() for sync client")
        response = await self._retry_request_async(
            "post",
            f"{self.base_url}/embed/batch",
            json={"texts": texts}
        )
        response.raise_for_status()
        data = response.json()
        embeddings = np.array(data["embeddings"], dtype=np.float32)
        return embeddings

    def health_check(self) -> bool:
        if self.use_async:
            raise RuntimeError("Use health_check_async() for async client")
        try:
            response = self._retry_request("get", f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False

    async def health_check_async(self) -> bool:
        if not self.use_async:
            raise RuntimeError("Use health_check() for sync client")
        try:
            response = await self._retry_request_async("get", f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False

    def close(self):
        if self.use_async:
            raise RuntimeError("Use close_async() for async client")
        self._client.close()

    async def close_async(self):
        if not self.use_async:
            raise RuntimeError("Use close() for sync client")
        await self._client.aclose()


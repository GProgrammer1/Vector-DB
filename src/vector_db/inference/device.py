"""Device management utilities for CPU/GPU flexibility."""

from typing import Any, Dict, Literal, Optional

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]


DeviceType = Literal["cpu", "cuda", "mps", "auto"]


def get_device(device: Optional[DeviceType] = "auto") -> str:
    
    if not TORCH_AVAILABLE:
        return "cpu"

    if device == "cpu":
        return "cpu"

    if device == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if device == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # Auto mode: select best available device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def is_gpu_available() -> bool:
   
    if not TORCH_AVAILABLE:
        return False
    cuda_available = torch.cuda.is_available()
    mps_available = (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    return cuda_available or mps_available


def get_device_info() -> Dict[str, Any]:
    """
    Returns:
        - device
        - cuda_available
        - mps_available
        - cuda_device_count
        - cuda_device_name (if available)
    """
    info: Dict[str, Any] = {
        "device": get_device(),
        "cuda_available": False,
        "mps_available": False,
        "cuda_device_count": 0,
        "cuda_device_name": None,
    }

    if not TORCH_AVAILABLE:
        return info

    info["cuda_available"] = torch.cuda.is_available()
    if info["cuda_available"]:
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)

    if hasattr(torch.backends, "mps"):
        info["mps_available"] = torch.backends.mps.is_available()

    return info


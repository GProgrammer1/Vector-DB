"""Device management utilities for CPU/GPU flexibility."""

from typing import Any, Literal, Optional

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]


DeviceType = Literal["cpu", "cuda", "mps", "auto"]


def get_device(device: Optional[DeviceType] = "auto") -> str:
    """
    Get the appropriate device string for PyTorch operations.

    Args:
        device: Device preference. Options:
            - "auto": Automatically select the best available device
            - "cpu": Force CPU usage
            - "cuda": Use CUDA if available, otherwise CPU
            - "mps": Use Apple Metal Performance Shaders if available (Apple Silicon)

    Returns:
        Device string ("cpu", "cuda", or "mps")

    Examples:
        >>> device = get_device()  # Auto-detect
        >>> device = get_device("cpu")  # Force CPU
        >>> device = get_device("cuda")  # Use GPU if available
    """
    if not TORCH_AVAILABLE:
        return "cpu"

    if device == "cpu":
        return "cpu"

    if device == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if device == "mps":
        # Apple Silicon GPU support
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
    """
    Check if GPU (CUDA or MPS) is available.

    Returns:
        True if GPU is available, False otherwise
    """
    if not TORCH_AVAILABLE:
        return False
    cuda_available = torch.cuda.is_available()
    mps_available = (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    return cuda_available or mps_available


def get_device_info() -> dict[str, Any]:
    """
    Get information about available devices.

    Returns:
        Dictionary with device information including:
        - device: Current device string
        - cuda_available: Whether CUDA is available
        - mps_available: Whether MPS (Apple Silicon) is available
        - cuda_device_count: Number of CUDA devices
        - cuda_device_name: Name of CUDA device (if available)
    """
    info: dict[str, Any] = {
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


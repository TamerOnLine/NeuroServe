from __future__ import annotations

import os
import time
import torch

DEVICE = os.getenv("DEVICE", "cuda:0")

def pick_device() -> torch.device:
    """Pick a valid CUDA device or fall back to CPU."""
    if torch.cuda.is_available():
        try:
            torch.cuda.get_device_properties(int(DEVICE.split(":")[1]))
            return torch.device(DEVICE)
        except Exception:
            return torch.device("cuda:0")
    return torch.device("cpu")

def pick_dtype(device: str | None = None) -> torch.dtype:
    """
    Pick an appropriate dtype based on device.
    - CUDA: bfloat16 if supported, otherwise float16
    - CPU: float32
    """
    if device is None:
        device = str(pick_device())

    if device.startswith("cuda") and torch.cuda.is_available():
        try:
            # Ampere (8.0) وما فوق يدعم bfloat16
            major, _ = torch.cuda.get_device_capability(0)
            if major >= 8:
                return torch.bfloat16
        except Exception:
            pass
        return torch.float16
    return torch.float32

def cuda_info() -> dict:
    """Retrieve CUDA device information."""
    info = {
        "torch_version": torch.__version__,
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "cuda_available": torch.cuda.is_available(),
        "device": str(pick_device()),
    }
    if torch.cuda.is_available():
        dev = pick_device()
        idx = dev.index or 0
        props = torch.cuda.get_device_properties(idx)
        info.update({
            "gpu_name": props.name,
            "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
            "device_index": idx,
            "driver": torch.cuda.driver_version if hasattr(torch.cuda, "driver_version") else None,
        })
    return info

def warmup() -> dict:
    """Run a small matrix multiplication on the GPU to ensure readiness."""
    dev = pick_device()
    size = int(os.getenv("WARMUP_MATMUL_SIZE", "1024"))
    x = torch.randn(size, size, device=dev)
    y = torch.randn(size, size, device=dev)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    z = x @ y
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    dt = time.time() - t0
    return {"shape": list(z.shape), "elapsed_sec": round(dt, 4), "device": str(dev)}

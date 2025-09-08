from __future__ import annotations
import os
import time
from pathlib import Path
import torch

# --- مسارات كاش ديناميكية داخل المشروع ---
BASE_DIR = Path(__file__).resolve().parent.parent  # gpu-server/
HF_HOME = BASE_DIR / "models_cache" / "huggingface"
TORCH_HOME = BASE_DIR / "models_cache" / "torch"

os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("TORCH_HOME", str(TORCH_HOME))
os.environ.setdefault("TRANSFORMERS_CACHE", str(HF_HOME / "hub"))
# اختياري: هدوء في اللوجز
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

DEVICE = os.getenv("DEVICE", "cuda:0")

def pick_device() -> torch.device:
    """Pick a valid device based on DEVICE env; respect 'cpu' even if CUDA exists."""
    # لو المستخدم اختار CPU صراحة
    if DEVICE.lower().startswith("cpu"):
        return torch.device("cpu")

    if torch.cuda.is_available():
        # DEVICE قد يكون "cuda" أو "cuda:1" الخ...
        try:
            idx = 0
            if ":" in DEVICE:
                idx = int(DEVICE.split(":")[1])
            torch.cuda.get_device_properties(idx)  # تأكيد صلاحية الفهرس
            return torch.device(f"cuda:{idx}")
        except Exception:
            return torch.device("cuda:0")
    return torch.device("cpu")

def pick_dtype(device: str | None = None) -> torch.dtype:
    """
    Pick an appropriate dtype based on the *selected* device.
    - CUDA: bfloat16 if supported, otherwise float16
    - CPU: float32
    """
    dev = torch.device(device) if device else pick_device()
    if dev.type == "cuda" and torch.cuda.is_available():
        try:
            # افحص قدرات نفس الكرت المختار
            idx = dev.index or 0
            major, _ = torch.cuda.get_device_capability(idx)
            if major >= 8:  # Ampere+ supports bfloat16
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
            "driver": getattr(torch.cuda, "driver_version", None),
        })
    return info

def warmup() -> dict:
    """Run a small matrix multiplication on the selected device."""
    dev = pick_device()
    size = int(os.getenv("WARMUP_MATMUL_SIZE", "1024"))
    x = torch.randn(size, size, device=dev)
    y = torch.randn(size, size, device=dev)

    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    z = x @ y
    if dev.type == "cuda":
        torch.cuda.synchronize()

    dt = time.time() - t0
    return {"shape": list(z.shape), "elapsed_sec": round(dt, 4), "device": str(dev)}

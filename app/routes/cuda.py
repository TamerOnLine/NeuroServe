import os
import time

import torch
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/cuda")
def cuda():
    return {
        "available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
        "device_name": torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else None,
    }


# alias لـ /cuda
@router.get("/cuda_info")
def cuda_info_alias():
    return cuda()


@router.post("/matmul")
def matmul(n: int = 1024):
    """Matrix multiplication test for performance benchmarking (CUDA or CPU fallback)."""
    force_cpu = os.getenv("FORCE_CPU", "0") == "1"
    use_cuda = torch.cuda.is_available() and not force_cpu
    device = torch.device("cuda" if use_cuda else "cpu")

    a = torch.rand((n, n), device=device)
    b = torch.rand((n, n), device=device)

    if use_cuda:
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        torch.matmul(a, b)
    if use_cuda:
        torch.cuda.synchronize()
    t1 = time.time()

    return {
        "n": n,
        "elapsed_ms": round((t1 - t0) * 1000, 3),
        "device": str(device),
        "cuda": use_cuda,
    }

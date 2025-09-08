from __future__ import annotations

import json
import os
import platform
import time
from typing import Literal

import psutil
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from .runtime import cuda_info, warmup, pick_device
from .toy_model import load_model

from fastapi import UploadFile, File, HTTPException
from .plugins.loader import discover, get, all_meta


load_dotenv()  # Read .env file if it exists

app = FastAPI(title="gpu_server", version="0.1.0")

# Global model and device
MODEL = None
DEVICE = None


@app.on_event("startup")
def _startup():
    """Load model and warm up on server startup."""
    global MODEL, DEVICE
    DEVICE = pick_device()
    MODEL, _ = load_model()
    warmup_result = warmup()
    print("[gpu_server] Warmup:", warmup_result)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/cuda")
def get_cuda():
    return cuda_info()


class MatmulReq(BaseModel):
    n: int = Field(1024, ge=1, le=8192, description="Matrix size N×N")


@app.post("/matmul")
def matmul(req: MatmulReq):
    n = req.n
    dev = DEVICE
    x = torch.randn(n, n, device=dev)
    y = torch.randn(n, n, device=dev)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    z = x @ y
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    return {"n": n, "device": str(dev), "elapsed_sec": round(time.time() - t0, 4)}


class InferReq(BaseModel):
    batch: int = Field(4, ge=1, le=256)
    in_features: Literal[512] = 512


@app.post("/infer")
def infer(req: InferReq):
    with torch.no_grad():
        x = torch.randn(req.batch, req.in_features, device=DEVICE)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        y = MODEL(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    return {
        "batch": req.batch,
        "out_shape": list(y.shape),
        "device": str(DEVICE),
        "elapsed_sec": round(time.time() - t0, 4),
    }


@app.get("/env")
def env(request: Request, pretty: bool = False):
    data = {
        **cuda_info(),
        "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
        "device_env": os.getenv("DEVICE", None),
    }
    if pretty:
        return Response(json.dumps(data, ensure_ascii=False, indent=2), media_type="application/json")
    return data


@app.get("/env/full")
def env_full(request: Request, pretty: bool = False):
    info = cuda_info()
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            gpus.append({
                "index": i,
                "name": p.name,
                "total_memory_gb": round(p.total_memory / (1024 ** 3), 2),
            })
    info.update({
        "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
        "device_env": os.getenv("DEVICE", None),
        "gpus": gpus,
    })
    if pretty:
        return Response(json.dumps(info, ensure_ascii=False, indent=2), media_type="application/json")
    return info


@app.get("/env/system")
def env_system(request: Request, pretty: bool = False):
    info = {
        **cuda_info(),
        "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
        "device_env": os.getenv("DEVICE", None),
        "os": platform.system(),
        "os_version": platform.version(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
    }
    if pretty:
        return Response(json.dumps(info, ensure_ascii=False, indent=2), media_type="application/json")
    return info


import subprocess

@app.post("/run/test-api")
def run_test_api():
    """Run the test_api.py script and return its output."""
    try:
        result = subprocess.run(
            ["python", "-m", "scripts.test_api"],
            capture_output=True,
            text=True,
            check=True
        )
        return {"success": True, "output": result.stdout}
    except subprocess.CalledProcessError as e:
        return {"success": False, "error": e.stderr}
    

@app.on_event("startup")
def _startup():
    global MODEL, DEVICE
    DEVICE = pick_device()
    MODEL, _ = load_model()  # TinyNet القديم يبقى كما هو (اختياري)  
    warmup_result = warmup()
    discover(reload=True)    # ← تحميل جميع الـ plugins عند البدء
    print("[gpu_server] Warmup:", warmup_result)


@app.get("/plugins")
def list_plugins():
    """قائمة المزوّدين المتاحين (من المجلد plugins/*)."""
    return {"providers": list(all_meta().values())}

@app.post("/inference")
def generic_inference(payload: dict):
    """
    استدعاء عام لأي نموذج.
    يتوقع payload يحوي على الأقل: {"provider": "<folder>", "task": "...", ...}
    """
    provider = str(payload.get("provider", "")).strip()
    if not provider:
        raise HTTPException(400, "Missing 'provider'")
    plugin = get(provider)
    if not plugin:
        raise HTTPException(404, f"Provider '{provider}' not found")
    try:
        out = plugin.infer(payload)
        return {"provider": provider, **out}
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}")




# Template rendering
templates = Jinja2Templates(directory="app/templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/ui", response_class=HTMLResponse)
def ui(request: Request):
    return templates.TemplateResponse("ui.html", {"request": request})


@app.get("/tools/model-size", response_class=HTMLResponse)
def model_size_page(request: Request):
    return templates.TemplateResponse("model_size.html", {"request": request})


@app.post("/api/model-size")
def api_model_size(
    in_features: int = Form(...),
    hidden: int = Form(...),
    out_features: int = Form(...),
    dtype: str = Form("fp32"),
):
    bytes_per = {"fp32": 4, "fp16": 2, "bf16": 2, "fp64": 8}.get(dtype, 4)
    weights = in_features * hidden + hidden * out_features
    memory_bytes = weights * bytes_per
    mem_mb = memory_bytes / (1024 ** 2)
    mem_gb = memory_bytes / (1024 ** 3)

    if mem_gb >= 1.0 or weights >= 50_000_000:
        recommendation = "GPU recommended"
    elif mem_mb <= 100 and weights <= 5_000_000:
        recommendation = "CPU suitable"
    else:
        recommendation = "Runs on CPU, faster on GPU"

    data = {
        "in_features": in_features,
        "hidden": hidden,
        "out_features": out_features,
        "total_weights": weights,
        "dtype": dtype,
        "bytes_per": bytes_per,
        "memory_MB": round(mem_mb, 2),
        "memory_GB": round(mem_gb, 3),
        "recommendation": recommendation,
    }
    return JSONResponse(data)


@app.get("/plugins/ui", response_class=HTMLResponse)
def plugins_console(request: Request):
    return templates.TemplateResponse("plugins.html", {"request": request})

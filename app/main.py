from __future__ import annotations

# ====== Standard Library ======
import json
import logging
import os
import platform
import subprocess
import sys
import threading  # ✅ جديد
import time
import uuid
from pathlib import Path
from typing import Literal

# ====== Third-Party ======
import psutil
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.routes.uploads import router as uploads_router

from .plugins.loader import all_meta, discover, get

# ====== Local Application ======
from .runtime import cuda_info, pick_device, warmup
from .toy_model import load_model

# (اختياري) unify
try:
    from app.utils.unify import unify_response

    _HAS_UNIFY = True
except Exception:
    _HAS_UNIFY = False

log = logging.getLogger("neuroserve")
logging.basicConfig(level=logging.INFO)

load_dotenv()  # Read .env file if it exists

app = FastAPI(title="gpu_server", version="0.1.0")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.include_router(uploads_router)

ENABLE_TRACE = os.getenv("TRACE_HTTP", "1").lower() in ("1", "true", "yes")

if ENABLE_TRACE:

    @app.middleware("http")
    async def tracing_middleware(request: Request, call_next):
        path = request.url.path
        if path.startswith("/static") or path.startswith("/plugins-data"):
            return await call_next(request)

        req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = req_id
        method = request.method
        t0 = time.perf_counter()
        log.info(f"[req {req_id}] -> {method} {path}")

        response = await call_next(request)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        response.headers["X-Request-ID"] = req_id
        log.info(f"[req {req_id}] <- {method} {path} | {response.status_code} | {elapsed_ms:.1f} ms")
        return response


# Global model and device
MODEL = None
DEVICE = None

BASE_DIR = os.path.dirname(__file__)
plugins_dir = os.path.join(BASE_DIR, "plugins")
app.mount("/plugins-data", StaticFiles(directory=plugins_dir), name="plugins-data")

# ✅ CI flag
CI_LIGHT_MODE = os.getenv("CI_LIGHT_MODE", "0").lower() in ("1", "true", "yes")


def _load_heavy():
    """Load heavy model + warmup in background"""
    global MODEL
    MODEL, _ = load_model()
    wr = warmup()
    print("[gpu_server] Warmup:", wr)


@app.on_event("startup")
def _startup():
    """Choose device, discover plugins quickly, and (optionally) load heavy stuff."""
    global MODEL, DEVICE
    DEVICE = pick_device()
    discover(reload=True)  # fast
    if CI_LIGHT_MODE:
        print("[gpu_server] CI_LIGHT_MODE=on (skip heavy load)")
    else:
        threading.Thread(target=_load_heavy, daemon=True).start()


@app.get("/plugins")
def list_plugins():
    meta = all_meta()
    names = sorted(meta.keys())
    return {"plugins": names, "meta": meta}


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

    _sync_if_cuda(dev)
    t0 = time.time()
    _ = x @ y
    _sync_if_cuda(dev)

    return {"n": n, "device": str(dev), "elapsed_sec": round(time.time() - t0, 4)}


class InferReq(BaseModel):
    batch: int = Field(4, ge=1, le=256)
    in_features: Literal[512] = 512


@app.post("/infer")
def infer(req: InferReq):
    dev = DEVICE
    with torch.no_grad():
        x = torch.randn(req.batch, req.in_features, device=dev)
        _sync_if_cuda(dev)
        t0 = time.time()
        y = MODEL(x)
        _sync_if_cuda(dev)

    return {
        "batch": req.batch,
        "out_shape": list(y.shape),
        "device": str(dev),
        "elapsed_sec": round(time.time() - t0, 4),
    }


# --- JSON sanitization helpers ---
def _to_jsonable(obj):
    try:
        import numpy as np
        import torch as _torch
    except Exception:
        np = None
        _torch = None

    if obj is None:
        return None
    if np is not None and isinstance(obj, (np.generic,)):
        return obj.item()
    if np is not None and hasattr(obj, "dtype") and hasattr(obj, "shape"):
        try:
            return obj.tolist()
        except Exception:
            return str(obj)
    if _torch is not None and hasattr(obj, "detach") and hasattr(obj, "cpu"):
        try:
            return obj.detach().cpu().tolist()
        except Exception:
            return str(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        t = type(obj)
        return t(_to_jsonable(v) for v in obj)
    if isinstance(obj, (str, int, float, bool)):
        return obj
    try:
        return str(obj)
    except Exception:
        return None


def _sync_if_cuda(dev):
    if hasattr(dev, "type") and dev.type == "cuda":
        torch.cuda.synchronize()


# باقي الـendpoints (env, inference, unified, templates...) مثل ملفك السابق
# لم ألمسها.


# --- helpers: env/info endpoints ---
@app.get("/env")
def env(request: Request, pretty: bool = False):
    data = {
        **cuda_info(),
        "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
        "device_env": os.getenv("DEVICE", None),
    }
    if pretty:
        return Response(
            json.dumps(data, ensure_ascii=False, indent=2),
            media_type="application/json",
        )
    return data


@app.get("/env/full")
def env_full(request: Request, pretty: bool = False):
    info = cuda_info()
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            gpus.append(
                {
                    "index": i,
                    "name": p.name,
                    "total_memory_gb": round(p.total_memory / (1024**3), 2),
                }
            )
    info.update(
        {
            "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
            "device_env": os.getenv("DEVICE", None),
            "gpus": gpus,
        }
    )
    if pretty:
        return Response(
            json.dumps(info, ensure_ascii=False, indent=2),
            media_type="application/json",
        )
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
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
    }
    if pretty:
        return Response(
            json.dumps(info, ensure_ascii=False, indent=2),
            media_type="application/json",
        )
    return info


@app.post("/run/test-api")
def run_test_api():
    """Run the test_api.py script and return its output."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "scripts.test_api"],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,  # seconds
        )
        return {"success": True, "output": result.stdout, "stderr": result.stderr}
    except subprocess.CalledProcessError as e:
        return {"success": False, "error": e.stderr or str(e)}
    except subprocess.TimeoutExpired as e:
        return {"success": False, "error": f"Timed out after {e.timeout}s"}
    except Exception as e:
        return {"success": False, "error": repr(e)}


# ================================
# Defaults from manifest (NEW)
# ================================
def apply_manifest_defaults(provider: str, task: str, req: dict) -> dict:
    """
    يدمج قيم defaults من manifest لكل مزوّد/مهمة مع طلب المستخدم.
    أولوية الدمج: قيم الطلب > الافتراضات.
    """
    mf_path = os.path.join(plugins_dir, provider, "manifest.json")
    try:
        with open(mf_path, "r", encoding="utf-8") as f:
            mf = json.load(f)
    except Exception:
        mf = {}
    defs = (mf.get("defaults") or {}).get(task or "infer", {})
    # طلب المستخدم يتفوّق على الافتراضات
    merged = {**defs, **req}
    # تأكد من تعيين provider/task في الناتج النهائي
    merged.setdefault("provider", provider)
    merged.setdefault("task", task or "infer")
    return merged


# ================================
# Generic inference (RAW)
# ================================
@app.post("/inference", response_class=JSONResponse)
async def generic_inference(request: Request):
    """
    يقبل:
      - جسم JSON مفرد: { "provider": "...", "task": "...", ... }
      - أو مصفوفة JSON: [ {..}, {..}, ... ]
    ويعيد نتيجة مفردة أو مصفوفة على الترتيب.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")

    def run_one(item: dict):
        provider = str(item.get("provider", "")).strip()
        if not provider:
            return {"error": "Missing 'provider'", "request": item}
        plugin = get(provider)
        if not plugin:
            return {"error": f"Provider '{provider}' not found", "request": item}
        try:
            # ✅ دمج الافتراضات من manifest قبل الاستدعاء
            task = str(item.get("task", "infer"))
            item_merged = apply_manifest_defaults(provider, task, item)

            out = plugin.infer(item_merged)
            if hasattr(out, "__await__"):
                # في حال كان plugin async
                return None, out  # نُعيد كوروتين ونعالجه خارجيًا
            if not isinstance(out, dict):
                out = {"result": out}
            data = {"provider": provider, **out}
            return _to_jsonable(data)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}", "request": item}

    # دفعة متعددة
    if isinstance(body, list):
        results = []
        for item in body:
            if not isinstance(item, dict):
                results.append({"error": "Each item must be an object.", "request": item})
                continue
            res = run_one(item)
            if isinstance(res, tuple) and res[1] is not None:
                out = await res[1]
                if not isinstance(out, dict):
                    out = {"result": out}
                results.append(_to_jsonable({"provider": item.get("provider"), **out}))
            else:
                results.append(res)
        return JSONResponse({"results": results})

    # طلب مفرد
    if isinstance(body, dict):
        res = run_one(body)
        if isinstance(res, tuple) and res[1] is not None:
            out = await res[1]
            if not isinstance(out, dict):
                out = {"result": out}
            return JSONResponse(_to_jsonable({"provider": body.get("provider"), **out}))
        return JSONResponse(res)

    raise HTTPException(status_code=400, detail="Body must be an object or a list of objects.")


# ================================
# Unified inference (ENVELOPE)
# ================================
@app.post("/api/unified-inference", response_class=JSONResponse)
async def unified_inference(request: Request):
    """
    نفس /inference لكن يُرجع Envelope موحّد:
    { provider, task, status, data, error?, meta?, elapsed_sec?, schema_version }
    يقبل جسم مفرد أو مصفوفة.
    """
    if not _HAS_UNIFY:
        raise HTTPException(
            status_code=501,
            detail="Unified response not available (app.utils.unify missing).",
        )

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body.")

    async def run_one(item: dict):
        provider = str(item.get("provider", "")).strip()
        task = str(item.get("task", "infer")).strip()
        if not provider or not task:
            return {
                "provider": provider or None,
                "task": task or None,
                "status": "error",
                "error": {"type": "BadRequest", "message": "Missing provider/task"},
                "schema_version": 1,
            }

        plugin = get(provider)
        if not plugin:
            return {
                "provider": provider,
                "task": task,
                "status": "error",
                "error": {"type": "ProviderNotFound", "message": f"Provider '{provider}' not found"},
                "schema_version": 1,
            }

        # ✅ دمج الافتراضات من manifest قبل الاستدعاء
        item_merged = apply_manifest_defaults(provider, task, item)

        out = plugin.infer(item_merged)
        if hasattr(out, "__await__"):
            out = await out
        if not isinstance(out, dict):
            out = {"result": out}
        # لفّ الاستجابة في ظرف موحّد
        return unify_response(provider, task, out)

    if isinstance(body, list):
        results = []
        for item in body:
            if not isinstance(item, dict):
                results.append(
                    {
                        "status": "error",
                        "error": {"type": "BadRequest", "message": "Each item must be an object"},
                        "schema_version": 1,
                    }
                )
                continue
            results.append(await run_one(item))
        return JSONResponse({"results": results})

    if isinstance(body, dict):
        return JSONResponse(await run_one(body))

    raise HTTPException(status_code=400, detail="Body must be an object or a list of objects.")


# Template rendering (resolve path robustly)
_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


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
    mem_mb = memory_bytes / (1024**2)
    mem_gb = memory_bytes / (1024**3)

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


@app.get("/infer-client", response_class=HTMLResponse)
async def infer_client(request: Request):
    return templates.TemplateResponse("infer-client.html", {"request": request})


@app.get("/plugins/unified", response_class=HTMLResponse)
def plugins_unified(request: Request):
    return templates.TemplateResponse("unified.html", {"request": request})

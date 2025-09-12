# app/routes/misc.py
import json
import os
import platform
import sys
import time
from pathlib import Path

import torch
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parents[1]  # -> app/
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# --------- صفحات HTML الموجودة عندك ---------
@router.get("/ui", response_class=HTMLResponse)
def ui_page(request: Request):
    # يحمّل templates/ui.html
    return templates.TemplateResponse("ui.html", {"request": request, "title": "UI"})


@router.get("/plugins/ui", response_class=HTMLResponse)
def plugins_console(request: Request):
    # يحمّل templates/plugins.html
    return templates.TemplateResponse("plugins.html", {"request": request, "title": "Plugins Console"})


@router.get("/unified", response_class=HTMLResponse)
@router.get("/plugins/unified", response_class=HTMLResponse)
def unified_page(request: Request):
    # يحمّل templates/unified.html
    return templates.TemplateResponse("unified.html", {"request": request, "title": "Unified"})


@router.get("/infer-client", response_class=HTMLResponse)
def infer_client_page(request: Request):
    # يحمّل templates/infer-client.html (بشرطة)
    return templates.TemplateResponse("infer-client.html", {"request": request, "title": "Infer Client"})


@router.get("/model-size", response_class=HTMLResponse)
def model_size_page(request: Request):
    # يحمّل templates/model_size.html (بـ underscore)
    return templates.TemplateResponse("model_size.html", {"request": request, "title": "Model Size"})


# --------- JSON معلومات بيئة ونظام ---------


def _cuda_brief() -> dict:
    avail = torch.cuda.is_available()
    cnt = torch.cuda.device_count() if avail else 0
    devs = []
    if avail:
        for i in range(cnt):
            p = torch.cuda.get_device_properties(i)
            devs.append(
                {
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "capability": tuple(torch.cuda.get_device_capability(i)),
                    "total_memory": int(p.total_memory),
                }
            )
    return {
        "cuda_available": avail,
        "device_count": cnt,
        "devices": devs,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }


@router.get("/env")
def env_brief():
    data = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cuda": _cuda_brief(),
    }
    return Response(
        content=json.dumps(data, indent=2, ensure_ascii=False),
        media_type="application/json",
    )


@router.get("/env/system")
def env_system():
    data = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python": sys.version,
        },
        "cpu_count": os.cpu_count(),
        "cwd": os.getcwd(),
    }
    return Response(
        content=json.dumps(data, indent=2, ensure_ascii=False),
        media_type="application/json",
    )


@router.get("/env/full")
def env_full():
    data = {"environ": dict(os.environ)}
    return Response(
        content=json.dumps(data, indent=2, ensure_ascii=False),
        media_type="application/json",
    )

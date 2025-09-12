# app/routes/infer.py
from __future__ import annotations

import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.plugins.loader import discover, get

router = APIRouter(tags=["infer"])


class InferRequest(BaseModel):
    provider: Optional[str] = None  # مثال: "ner"
    task: Optional[str] = None
    text: Optional[str] = None  # الصيغة المختصرة
    payload: Dict[str, Any] = {}  # الحمولة الكاملة


@router.post("/infer")
async def infer(body: InferRequest):  # ← لاحظ: body بدل Request
    # دعم الصيغة المختصرة: {"text": "..."} → provider="ner"
    provider = body.provider
    payload: Dict[str, Any] = body.payload or {}

    if not provider and body.text:
        provider = "ner"
        payload = {"text": body.text}

    if not provider:
        raise HTTPException(
            400,
            "Missing 'provider'. Use {'provider':'ner','payload':{'text':'...'}} " "or the short form {'text':'...'}.",
        )

    if body.task is not None and "task" not in payload:
        payload["task"] = body.task

    plugin = get(provider) or (discover() or get(provider))
    if plugin is None:
        raise HTTPException(404, f"Provider '{provider}' not found")

    t0 = time.time()
    try:
        out = await plugin.infer(payload)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"{type(e).__name__}: {e}")
    elapsed_ms = round((time.time() - t0) * 1000.0, 3)

    return {
        "status": "ok",
        "provider": provider,
        "task": payload.get("task"),
        "out": out,
        "meta": {"elapsed_ms": elapsed_ms},
        "schema_version": 1,
    }

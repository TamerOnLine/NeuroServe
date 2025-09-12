# app/utils/unify.py
from copy import deepcopy
from typing import Any, Dict, Optional

SCHEMA_VERSION = 1


def _jsonable(x: Any):
    """حوّل الأنواع الشائعة (numpy/torch) إلى JSON صالح."""
    try:
        import numpy as np
    except Exception:
        np = None
    try:
        import torch
    except Exception:
        torch = None

    if x is None:
        return None
    if np is not None and isinstance(x, (np.generic,)):
        return x.item()
    if np is not None and hasattr(x, "dtype") and hasattr(x, "shape"):
        try:
            return x.tolist()
        except Exception:
            return str(x)
    if torch is not None and hasattr(x, "detach") and hasattr(x, "cpu"):
        try:
            return x.detach().cpu().tolist()
        except Exception:
            return str(x)
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        t = type(x)
        return t(_jsonable(v) for v in x)
    if isinstance(x, (str, int, float, bool)):
        return x
    try:
        return str(x)
    except Exception:
        return None


def is_already_unified(raw: Dict[str, Any]) -> bool:
    return isinstance(raw, dict) and raw.get("schema_version") is not None and raw.get("status") in ("ok", "error")


def unify_response(provider: str, task: str, raw: Any, request_id: Optional[str] = None) -> Dict[str, Any]:
    """غلاف موحّد لنتائج المزوّدين مع دعم ميتاداتا وتنظيف JSON."""
    # طبّع النتيجة غير القابلة للتسلسل
    if not isinstance(raw, dict):
        raw = {"result": raw}

    # حالة: عندك status موجود لكن بدون schema_version → كمّله وأرجعه
    if isinstance(raw, dict) and raw.get("status") in ("ok", "error") and raw.get("schema_version") is None:
        out = deepcopy(raw)
        out.setdefault("provider", provider)
        out.setdefault("task", task)
        out["schema_version"] = SCHEMA_VERSION
        if request_id:
            out.setdefault("meta", {})
            out["meta"]["request_id"] = request_id
        return _jsonable(out)

    # حالة: موحّد أصلاً
    if is_already_unified(raw):
        out = deepcopy(raw)
        out.setdefault("provider", provider)
        out.setdefault("task", task)
        if request_id:
            out.setdefault("meta", {})
            out["meta"]["request_id"] = request_id
        return _jsonable(out)

    # اجمع ميتاداتا شائعة (أضفنا input/usage)
    meta_keys = ("device", "model", "backend", "params", "input", "usage", "input_chars", "truncated_to_1024_tokens")
    meta = {k: raw.get(k) for k in meta_keys if k in raw}

    # مسار الخطأ (قديم)
    if "error" in raw:
        err = raw["error"]
        if not isinstance(err, dict):
            err = {"type": "Error", "message": str(err)}
        out = {
            "provider": provider,
            "task": task,
            "status": "error",
            "error": _jsonable(err),
            "schema_version": SCHEMA_VERSION,
        }
        if request_id or meta:
            out["meta"] = _jsonable({**meta, **({"request_id": request_id} if request_id else {})}) or None
        return out

    # مسار النجاح
    out = {
        "provider": provider,
        "task": task,
        "status": "ok",
        "elapsed_sec": raw.get("elapsed_sec"),
        "data": _jsonable(raw),  # أبقِ الشكل القديم كما هو داخل data
        "schema_version": SCHEMA_VERSION,
    }
    if request_id or meta:
        out["meta"] = _jsonable({**meta, **({"request_id": request_id} if request_id else {})}) or None
    return out

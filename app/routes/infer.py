# app/routes/infer.py
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class InferIn(BaseModel):
    text: str | None = ""


@router.post("/infer")
def infer(body: InferIn):
    """
    Safe fallback inference that always returns HTTP 200 (CI-friendly).
    No heavy models; just echoes and simple stats.
    """
    text = (body.text or "").strip()
    words = len(text.split()) if text else 0
    chars = len(text)

    return {
        "ok": True,
        "echo": text,
        "words": words,
        "chars": chars,
    }

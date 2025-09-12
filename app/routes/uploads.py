import io
import re

import numpy as np
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

router = APIRouter()


# ============== Utilities ==============
def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def summarize_text_simple(text: str, max_sentences: int = 3) -> str:
    sentences = re.split(r"(?<=[.!?]) +", text)
    return " ".join(sentences[:max_sentences])


def load_image_bytes(content: bytes) -> Image.Image:
    return Image.open(io.BytesIO(content))


def preprocess_image(img: Image.Image, max_size: int = 512) -> Image.Image:
    img = img.convert("RGB")
    img.thumbnail((max_size, max_size))
    return img


def image_to_numpy(img: Image.Image) -> np.ndarray:
    return np.array(img)


# ============== Endpoints ==============
@router.post("/text")
async def upload_text(
    file: UploadFile = File(None),
    text: str = Form(None),
    summarize: bool = Form(False),
    max_sentences: int = Form(3),
):
    if file:
        content = await file.read()
        text = content.decode("utf-8")

    if not text:
        return JSONResponse({"error": "No text provided"}, status_code=400)

    pre = clean_text(text)
    post = summarize_text_simple(pre, max_sentences) if summarize else pre

    return JSONResponse({"ok": True, "length": len(pre), "preview": post[:1000], "summarized": summarize})


@router.post("/image")
async def upload_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        return JSONResponse({"error": "Invalid file type"}, status_code=400)

    content = await file.read()
    img = load_image_bytes(content)
    img = preprocess_image(img)

    arr = image_to_numpy(img)
    h, w, c = arr.shape

    return JSONResponse(
        {"ok": True, "filename": file.filename, "size": len(content), "width": w, "height": h, "channels": c}
    )

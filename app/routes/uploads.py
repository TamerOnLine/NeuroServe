# app/routes/uploads.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, List
import io

# Text processing imports
import re

# Image processing imports
from PIL import Image
import numpy as np

router = APIRouter(prefix="/upload", tags=["upload"])

# ---- Helpers ----
def clean_text(text: str) -> str:
    """Basic pre-processing: normalize newlines/whitespace, strip BOM, collapse spaces."""
    if text is None:
        return ""
    # Strip BOM if present
    text = text.replace("\ufeff", "")
    # Normalize newlines and spaces
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def summarize_text_simple(text: str, max_sentences: int = 3) -> str:
    """Very tiny 'post-processing' summarizer: take first N sentences. (No ML dependency)"""
    # naive split by punctuation
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(sentences[:max_sentences]).strip()

def load_image_bytes(img_bytes: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

def preprocess_image(img: Image.Image, size: int = 224) -> Image.Image:
    """Resize & center-crop to square 'size'."""
    w, h = img.size
    # center-crop to square
    min_side = min(w, h)
    left = (w - min_side) // 2
    top = (h - min_side) // 2
    img = img.crop((left, top, left + min_side, top + min_side))
    # resize
    img = img.resize((size, size))
    return img

def image_to_numpy(img: Image.Image) -> List[List[List[int]]]:
    """Convert to nested list (uint8) just for returning small previews."""
    arr = np.array(img, dtype=np.uint8)
    # Beware of huge payloads; truncate preview if needed
    if arr.shape[0] > 256 or arr.shape[1] > 256:
        arr = arr[:256, :256, :]
    return arr.tolist()

# ---- Endpoints ----
@router.post("/text")
async def upload_text(
    text: Optional[str] = Form(default=None),
    file: Optional[UploadFile] = File(default=None, description="Optional .txt file"),
    summarize: bool = Form(default=False),
    max_sentences: int = Form(default=3),
):
    """
    Accepts raw text via form field or a .txt file upload.
    Pre-processing: whitespace normalization.
    Post-processing (optional): naive summarization by first N sentences.
    """
    content = ""
    if text:
        content = text
    elif file is not None:
        if not file.filename.lower().endswith(".txt"):
            raise HTTPException(status_code=400, detail="Please upload a .txt file")
        content = (await file.read()).decode("utf-8", errors="replace")
    else:
        raise HTTPException(status_code=400, detail="Provide 'text' or a .txt 'file'")

    pre = clean_text(content)
    post = summarize_text_simple(pre, max_sentences) if summarize else pre
    return JSONResponse({"ok": True, "length": len(pre), "preview": post[:1000], "summarized": summarize})

@router.post("/image")
async def upload_image(
    image: UploadFile = File(..., description="Image file: jpeg/png/webp"),
    size: int = Form(default=224),
    return_preview: bool = Form(default=True),
):
    """
    Accepts an image file, applies basic pre-processing (center-crop + resize).
    Optionally returns a small pixel preview (uint8) to verify pipeline.
    """
    name = image.filename.lower()
    if not any(name.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp")):
        raise HTTPException(status_code=400, detail="Supported: .jpg, .jpeg, .png, .webp")

    data = await image.read()
    img = load_image_bytes(data)
    pre = preprocess_image(img, size=size)

    payload = {"ok": True, "size": size, "width": pre.width, "height": pre.height}
    if return_preview:
        payload["preview_rgb"] = image_to_numpy(pre)  # beware size

    return JSONResponse(payload)

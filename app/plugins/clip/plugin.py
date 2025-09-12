from __future__ import annotations

import time
import traceback
from io import BytesIO
from pathlib import Path

import requests
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from app.plugins.base import AIPlugin
from app.runtime import pick_device, pick_dtype


class Plugin(AIPlugin):
    tasks = ["embed-text", "embed-image", "similarity"]

    def load(self) -> None:
        self.dev = pick_device()
        self.model_name = "openai/clip-vit-base-patch32"

        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = (
            CLIPModel.from_pretrained(self.model_name, low_cpu_mem_usage=True, dtype=pick_dtype(str(self.dev)))
            .to(self.dev)
            .eval()
        )

        print("[plugin] clip loaded on", self.dev)

    def _load_image(self, image_ref: str) -> Image.Image:
        """يدعم تحميل صورة من URL أو مسار محلي"""
        try:
            if image_ref.startswith("http://") or image_ref.startswith("https://"):
                resp = requests.get(image_ref, timeout=20)
                resp.raise_for_status()
                return Image.open(BytesIO(resp.content)).convert("RGB")
            else:
                return Image.open(Path(image_ref)).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image: {e}")

    def infer(self, payload: dict) -> dict:
        task = payload.get("task")
        if task not in self.tasks:
            return {"task": task, "error": f"Unsupported task for CLIP: {task}"}

        try:
            if self.dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()

            if task == "embed-text":
                text = payload.get("text") or payload.get("input")
                if not text:
                    return {"task": task, "error": "text is required"}
                inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.dev)
                with torch.no_grad():
                    emb = self.model.get_text_features(**inputs)
                out = emb.cpu().tolist()

            elif task == "embed-image":
                img_ref = payload.get("image_url") or payload.get("input")
                if not img_ref:
                    return {"task": task, "error": "image_url is required"}
                img = self._load_image(img_ref)
                inputs = self.processor(images=img, return_tensors="pt").to(self.dev)
                with torch.no_grad():
                    emb = self.model.get_image_features(**inputs)
                out = emb.cpu().tolist()

            elif task == "similarity":
                text = payload.get("text")
                img_ref = payload.get("image_url")
                if not text or not img_ref:
                    return {"task": task, "error": "text and image_url are required"}
                img = self._load_image(img_ref)
                inputs = self.processor(text=[text], images=img, return_tensors="pt", padding=True).to(self.dev)
                with torch.no_grad():
                    logits_per_image = self.model(**inputs).logits_per_image
                    sim = logits_per_image.softmax(dim=-1)[0].cpu().tolist()
                out = sim

            if self.dev.type == "cuda":
                torch.cuda.synchronize()
            elapsed = round(time.time() - t0, 3)

            return {
                "task": task,
                "device": str(self.dev),
                "model": self.model_name,
                "output": out,
                "elapsed_sec": elapsed,
            }

        except Exception as e:
            return {"task": task, "error": str(e), "traceback": traceback.format_exc()}

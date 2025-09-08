from __future__ import annotations
import io, time, requests
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import models
from app.plugins.base import AIPlugin
from app.runtime import pick_device

class Plugin(AIPlugin):
    tasks = ["classify"]

    def load(self) -> None:
        self.dev = pick_device()
        weights = models.ResNet18_Weights.DEFAULT
        self.model = models.resnet18(weights=weights).to(self.dev).eval()
        self.preprocess = weights.transforms()
        # أسماء الأصناف (ImageNet)
        self.labels = weights.meta.get("categories", [])
        # warm
        with torch.no_grad():
            _ = self.model(torch.zeros(1, 3, 224, 224, device=self.dev))
        print("[plugin] resnet18 loaded on", self.dev)

    def infer(self, payload: dict) -> dict:
        url = str(payload.get("image_url", "")).strip()
        topk = int(payload.get("topk", 5))
        if not url:
            raise ValueError("image_url is required")

        # حمل الصورة
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")

        # تحضير + استدلال
        x = self.preprocess(img).unsqueeze(0).to(self.dev)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            logits = self.model(x)
        if torch.cuda.is_available(): torch.cuda.synchronize()

        probs = F.softmax(logits, dim=1)[0]
        p, idx = probs.topk(topk)
        classes = [self.labels[i] if i < len(self.labels) else str(i) for i in idx.tolist()]

        return {
            "task": "classify",
            "device": str(self.dev),
            "topk": topk,
            "classes": classes,
            "probs": [float(v) for v in p.tolist()],
            "elapsed_sec": round(time.time() - t0, 4),
        }

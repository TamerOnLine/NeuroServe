from __future__ import annotations
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from app.plugins.base import AIPlugin
from app.runtime import pick_device

class Plugin(AIPlugin):
    tasks = ["text-classify"]

    def load(self) -> None:
        self.dev = pick_device()
        model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.dev).eval()
        print("[plugin] distilbert loaded on", self.dev)

    def infer(self, payload: dict) -> dict:
        # يقبل input / text
        text = (payload.get("text") or payload.get("input") or "").strip()
        if not text:
            return {"task": "text-classify", "error": "text is required"} 


        inputs = self.tokenizer(text, return_tensors="pt").to(self.dev)
        if self.dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            outputs = self.model(**inputs)
        if self.dev.type == "cuda":
            torch.cuda.synchronize()

        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        labels = ["negative", "positive"]
        score, idx = torch.max(probs, dim=-1)

        return {
            "task": "text-classify",
            "device": str(self.dev),
            "text": text,
            "label": labels[idx.item()],
            "confidence": round(score.item() * 100, 2),
            "probs": {labels[i]: round(p.item() * 100, 2) for i, p in enumerate(probs)},
            "elapsed_sec": round(time.time() - t0, 4)
        }

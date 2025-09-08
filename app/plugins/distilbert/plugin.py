from __future__ import annotations
import time
from typing import List, Union

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from app.plugins.base import AIPlugin
from app.runtime import pick_device


class Plugin(AIPlugin):
    tasks = ["text-classify"]

    def load(self) -> None:
        self.dev = pick_device()
        self.model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # مبدئيًا نخليها FP32 (أكثر ثباتًا). لو تبغى FP16 على CUDA غيّر use_fp16=True تحت.
        use_fp16 = False and getattr(self.dev, "type", "") == "cuda"
        dtype = torch.float16 if use_fp16 else torch.float32

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        ).to(self.dev, dtype=dtype).eval()

        # حفظ خريطة التصنيفات إن توفرت
        self.id2label = getattr(self.model.config, "id2label", {0: "NEGATIVE", 1: "POSITIVE"})
        self.label2id = {v: k for k, v in self.id2label.items()}

        print("[plugin] distilbert loaded on", self.dev)

    def _classify(self, texts: List[str]) -> dict:
        # Tokenize مع ضبط التجاوز/الحشو
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(self.dev)

        # قياس زمن الاستدلال
        if getattr(self.dev, "type", "") == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        with torch.no_grad():
            out = self.model(**enc)
            probs = torch.nn.functional.softmax(out.logits, dim=-1)

        if getattr(self.dev, "type", "") == "cuda":
            torch.cuda.synchronize()
        elapsed = round(time.time() - t0, 4)

        # بناء النتائج لكل عنصر
        results = []
        for i, p in enumerate(probs):
            score, idx = torch.max(p, dim=-1)
            label = self.id2label.get(idx.item(), str(idx.item()))
            results.append({
                "text": texts[i],
                "label": label.lower(),
                "confidence": round(score.item() * 100, 2),
                "probs": {
                    self.id2label.get(j, str(j)).lower(): round(p_j.item() * 100, 2)
                    for j, p_j in enumerate(p)
                }
            })

        return {
            "task": "text-classify",
            "device": str(self.dev),
            "model": self.model_name,
            "elapsed_sec": elapsed,
            "results": results,
        }

    def infer(self, payload: dict) -> dict:
        # يقبل input / text (string أو list)
        raw = payload.get("text") or payload.get("input")
        if raw is None:
            return {"task": "text-classify", "error": "text is required"}

        if isinstance(raw, str):
            texts = [raw.strip()]
        elif isinstance(raw, list):
            texts = [str(x).strip() for x in raw if str(x).strip()]
        else:
            return {"task": "text-classify", "error": "text must be a string or list of strings"}

        if not texts:
            return {"task": "text-classify", "error": "text is empty"}

        return self._classify(texts)

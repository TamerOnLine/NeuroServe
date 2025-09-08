# -*- coding: utf-8 -*-
from typing import List, Literal, Optional, Any, Dict
from pydantic import BaseModel, Field
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from fastapi import HTTPException

# ====== طلب/استجابة ======
class NERRequest(BaseModel):
    provider: Literal["ner"] = "ner"
    task: Literal["extract-entities"] = "extract-entities"
    text: str = Field(..., description="النص المراد استخراج الكيانات منه")
    aggregation: Literal["simple", "none", "first", "average", "max"] = "simple"
    model_override: Optional[str] = None

class Entity(BaseModel):
    start: int
    end: int
    word: str
    label: str
    score: float

class NERResponse(BaseModel):
    provider: Literal["ner"] = "ner"
    task: Literal["extract-entities"] = "extract-entities"
    device: str
    model: str
    entities: List[Entity]
    raw: Any = None  # اختياري: أزلْه لاحقًا لو تريد إخراجًا نظيفًا

# ====== Plugin ======
class Plugin:
    """
    Plugin: NER (Named Entity Recognition)
    - task: extract-entities
    """
    name = "ner"
    default_model = "dslim/bert-base-NER"

    def __init__(self, device: Optional[str] = None, model_name: Optional[str] = None):
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_name = model_name or self.default_model

        # تحميل النموذج والمعالج
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)

        # إعداد الـ pipeline
        device_index = -1
        if self.device.startswith("cuda") and torch.cuda.is_available():
            try:
                device_index = int(self.device.split(":")[1])
            except Exception:
                device_index = 0

        self.pipe = pipeline(
            "token-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device_index,
            aggregation_strategy="simple"
        )

    # تُستدعى من loader عند الإقلاع
    def load(self):
        return {"name": self.name, "device": self.device, "model": self.model_name}

    # نقطة الدخول الموحّدة
    async def infer(self, payload: Dict[str, Any]):
        try:
            # 1) لو وصلك manifest كامل بالغلط، فكّه لأقرب payload
            if "text" not in payload:
                if isinstance(payload.get("payload"), dict):
                    payload = payload["payload"]
                elif isinstance(payload.get("examples"), list) and payload["examples"]:
                    ex0 = payload["examples"][0]
                    if isinstance(ex0, dict) and isinstance(ex0.get("payload"), dict):
                        payload = ex0["payload"]

            # 2) اجلب النص من text أو مفاتيح بديلة
            text = payload.get("text")
            if text is None:
                if isinstance(payload.get("input"), str):
                    text = payload["input"]
                elif isinstance(payload.get("title"), str):
                    text = payload["title"]
                elif isinstance(payload.get("query"), str):
                    text = payload["query"]
                elif isinstance(payload.get("sentence"), str):
                    text = payload["sentence"]
                elif isinstance(payload.get("texts"), list) and payload["texts"]:
                    text = payload["texts"][0]

            if text is None or not isinstance(text, str) or not text.strip():
                raise HTTPException(422, "NER expects 'text' (or 'input'/'title'/'texts[0]').")

            # 3) حمولة منظّفة للـ schema
            norm = {
                "provider": "ner",
                "task": payload.get("task", "extract-entities"),
                "text": text,
                "aggregation": payload.get("aggregation", "simple"),
                "model_override": payload.get("model_override"),
            }

            req = NERRequest(**norm)
            return await self.extract_entities(req)

        except HTTPException:
            raise
        except Exception as e:
            import traceback; traceback.print_exc()
            raise HTTPException(500, f"{type(e).__name__}: {e}")

    async def extract_entities(self, req: NERRequest) -> Dict[str, Any]:
        # override مؤقت عند الطلب
        if req.model_override and req.model_override != self.model_name:
            temp_tok = AutoTokenizer.from_pretrained(req.model_override, use_fast=True)
            temp_mod = AutoModelForTokenClassification.from_pretrained(req.model_override)

            device_index = -1
            if self.device.startswith("cuda") and torch.cuda.is_available():
                try:
                    device_index = int(self.device.split(":")[1])
                except Exception:
                    device_index = 0

            temp_pipe = pipeline(
                "token-classification",
                model=temp_mod,
                tokenizer=temp_tok,
                device=device_index,
                aggregation_strategy=req.aggregation
            )
            out = temp_pipe(req.text)
            used_model = req.model_override
        else:
            # تبديل استراتيجية التجميع إن لزم (مقارنة آمنة)
            if req.aggregation != getattr(self.pipe, "aggregation_strategy", "simple"):
                device_index = -1
                if self.device.startswith("cuda") and torch.cuda.is_available():
                    try:
                        device_index = int(self.device.split(":")[1])
                    except Exception:
                        device_index = 0
                temp_pipe = pipeline(
                    "token-classification",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=device_index,
                    aggregation_strategy=req.aggregation
                )
                out = temp_pipe(req.text)
            else:
                out = self.pipe(req.text)
            used_model = self.model_name

        # توحيد الإخراج
        entities: List[Entity] = []
        for e in out:
            label = e.get("entity_group") or e.get("entity") or "UNK"
            entities.append(Entity(
                start=int(e["start"]),
                end=int(e["end"]),
                word=e["word"],
                label=str(label),
                score=float(e["score"])
            ))

        resp = NERResponse(
            device=self.device,
            model=used_model,
            entities=entities,
            raw=out
        )
        return resp.model_dump()

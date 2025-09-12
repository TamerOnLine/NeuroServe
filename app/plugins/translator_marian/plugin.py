from __future__ import annotations

import time
import traceback
from typing import Dict, Optional, Tuple

import torch
from transformers import MarianMTModel, MarianTokenizer

from app.plugins.base import AIPlugin
from app.runtime import pick_device, pick_dtype

_LANG_ALIASES = {
    "ar": "arabic",
    "arabic": "arabic",
    "عربي": "arabic",
    "العربية": "arabic",
    "en": "english",
    "english": "english",
    "انجليزي": "english",
    "الانجليزية": "english",
}
_MARIAN_MAP: Dict[Tuple[str, str], str] = {
    ("arabic", "english"): "Helsinki-NLP/opus-mt-ar-en",
    ("english", "arabic"): "Helsinki-NLP/opus-mt-en-ar",
}


def _norm_lang(x: str, dflt: str) -> str:
    if not x:
        return dflt
    return _LANG_ALIASES.get(str(x).strip().lower(), str(x).strip().lower())


class Plugin(AIPlugin):
    tasks = ["translate"]

    def load(self) -> None:
        self.dev = pick_device()
        self.dtype = pick_dtype(str(self.dev))
        self._cache: Dict[str, Tuple[MarianTokenizer, MarianMTModel]] = {}
        print("[plugin] translator_marian loaded on", self.dev)

    def _get(self, name: str) -> Tuple[MarianTokenizer, MarianMTModel]:
        if name in self._cache:
            return self._cache[name]
        tok = MarianTokenizer.from_pretrained(name)
        mdl = MarianMTModel.from_pretrained(name, dtype=self.dtype).to(self.dev).eval()
        self._cache[name] = (tok, mdl)
        return tok, mdl

    def _once(
        self,
        text: str,
        name: str,
        max_new: int,
        num_beams: int,
        do_sample: bool,
        temperature: Optional[float],
        top_p: Optional[float],
    ) -> str:
        tok, mdl = self._get(name)
        enc = tok(
            [text], return_tensors="pt", padding=True, truncation=True, max_length=512, return_attention_mask=True
        )
        enc = {k: v.to(self.dev) for k, v in enc.items()}
        gen = dict(
            max_new_tokens=max_new,
            num_beams=(1 if do_sample else max(1, num_beams)),
            do_sample=do_sample,
            early_stopping=True,
        )
        if do_sample:
            if temperature is not None:
                gen["temperature"] = float(temperature)
            if top_p is not None:
                gen["top_p"] = float(top_p)
        with torch.no_grad():
            out = mdl.generate(**enc, **gen)
        return tok.batch_decode(out, skip_special_tokens=True)[0].strip()

    def infer(self, payload: dict) -> dict:
        text = (payload.get("text") or payload.get("input") or "").strip()
        if not text:
            return {"task": "translate", "error": "text is required"}

        src = _norm_lang(payload.get("source_lang", ""), "arabic")
        tgt = _norm_lang(payload.get("target_lang", ""), "english")
        if (src, tgt) not in _MARIAN_MAP:
            return {"task": "translate", "error": "translator_marian supports only arabic↔english"}

        name = _MARIAN_MAP[(src, tgt)]
        max_new = max(8, min(int(payload.get("max_new_tokens", 64)), 256))
        num_beams = int(payload.get("num_beams", 4))
        do_sample = bool(payload.get("do_sample", False))
        temperature = float(payload.get("temperature", 0.7)) if do_sample else None
        top_p = float(payload.get("top_p", 0.9)) if do_sample else None

        try:
            if getattr(self.dev, "type", "") == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            out = self._once(text, name, max_new, num_beams, do_sample, temperature, top_p)
            if getattr(self.dev, "type", "") == "cuda":
                torch.cuda.synchronize()
            return {
                "task": "translate",
                "provider": "translator_marian",
                "device": str(self.dev),
                "model": name,
                "backend": "marian",
                "params": {
                    "source_lang": src,
                    "target_lang": tgt,
                    "max_new_tokens": max_new,
                    "do_sample": do_sample,
                    "num_beams": num_beams,
                    "temperature": temperature,
                    "top_p": top_p,
                },
                "input_chars": len(text),
                "output": out,
                "elapsed_sec": round(time.time() - t0, 3),
            }
        except Exception as e:
            return {"task": "translate", "error": str(e), "traceback": traceback.format_exc()}

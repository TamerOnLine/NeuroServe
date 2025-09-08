from __future__ import annotations
import time
import traceback
from typing import List

import torch
from transformers import AutoTokenizer, MT5ForConditionalGeneration

from app.plugins.base import AIPlugin
from app.runtime import pick_device, pick_dtype

# خرائط أسماء اللغات الشائعة ↔︎ الصياغة التي يفهمها mT5 في الـprefix
_LANG_ALIASES = {
    "ar": "arabic", "arabic": "arabic", "arabi": "arabic", "عربي": "arabic", "العربية": "arabic",
    "en": "english", "english": "english", "inglizi": "english", "انجليزي": "english", "الانجليزية": "english",
    "fr": "french", "français": "french", "فرنسي": "french",
    "de": "german", "deutsch": "german", "الالمانية": "german",
    "tr": "turkish", "turkish": "turkish", "تركي": "turkish",
    "es": "spanish", "spanish": "spanish", "اسباني": "spanish",
}
def _norm_lang(name: str, default: str) -> str:
    if not name:
        return default
    key = str(name).strip().lower()
    return _LANG_ALIASES.get(key, key)


def _soft_chunks(text: str, approx_len: int = 400) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks, cur = [], []
    cur_len = 0
    for w in words:
        wlen = len(w) + (1 if cur else 0)
        if cur_len + wlen > approx_len and cur:
            chunks.append(" ".join(cur))
            cur = [w]
            cur_len = len(w)
        else:
            cur.append(w)
            cur_len += wlen
    if cur:
        chunks.append(" ".join(cur))
    return chunks


class Plugin(AIPlugin):
    tasks = ["translate"]

    def load(self) -> None:
        self.dev = pick_device()
        self.model_name = "google/mt5-small"

        # ✅ أوقف تحذير fast tokenizer/legacy
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)

        # ✅ استخدم dtype الحديث + اختيار تلقائي يناسب الجهاز
        self.model = MT5ForConditionalGeneration.from_pretrained(
            self.model_name,
            low_cpu_mem_usage=True,
            dtype=pick_dtype(str(self.dev))
        ).to(self.dev)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        try:
            _ = self._translate_once("hello", "english", "arabic",
                                     max_new=16, do_sample=False, num_beams=4)
        except Exception as e:
            print("[plugin][mt5] warmup warn:", e)

        print("[plugin] mt5 loaded on", self.dev)

    def _translate_once(
        self,
        text: str,
        src: str,
        tgt: str,
        max_new: int,
        do_sample: bool,
        num_beams: int,
        temperature: float | None = None,
        top_p: float | None = None,
    ) -> str:
        prefix = f"translate {src} to {tgt}: "
        enc = self.tokenizer(
            prefix + text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        # انقل للـdevice، وحوّل dtype فقط للتنسورات العائمة
        enc = {
            k: (
                v.to(self.dev, dtype=self.model.dtype) if (torch.is_tensor(v) and v.is_floating_point())
                else (v.to(self.dev) if torch.is_tensor(v) else v)
            )
            for k, v in enc.items()
        }

        gen_kwargs = dict(
            max_new_tokens=max_new,
            do_sample=do_sample,
            num_beams=(1 if do_sample else max(1, num_beams)),
            length_penalty=1.0,
            early_stopping=True,
            no_repeat_ngram_size=3,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        if do_sample:
            if temperature is not None:
                gen_kwargs["temperature"] = float(temperature)
            if top_p is not None:
                gen_kwargs["top_p"] = float(top_p)

        with torch.no_grad():
            out_ids = self.model.generate(**enc, **gen_kwargs)

        return self.tokenizer.decode(out_ids[0], skip_special_tokens=True)

    def infer(self, payload: dict) -> dict:
        text = (payload.get("text") or payload.get("input") or "").strip()
        if not text:
            return {"task": "translate", "error": "text is required"}

        src = _norm_lang(str(payload.get("source_lang", "")), "arabic")
        tgt = _norm_lang(str(payload.get("target_lang", "")), "english")
        if src == tgt:
            return {"task": "translate", "error": "source_lang and target_lang must be different"}

        max_new = int(payload.get("max_new_tokens", 128))
        max_new = max(8, min(max_new, 256))

        do_sample = bool(payload.get("do_sample", False))
        num_beams = int(payload.get("num_beams", 4))

        temperature = float(payload.get("temperature", 0.7))
        top_p = float(payload.get("top_p", 0.9))

        use_chunking = bool(payload.get("chunking", True))
        chunk_len = int(payload.get("chunk_len", 400))
        parts = _soft_chunks(text, chunk_len) if use_chunking else [text]

        try:
            if getattr(self.dev, "type", "") == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()

            outs: List[str] = []
            for p in parts:
                outs.append(self._translate_once(
                    p, src, tgt, max_new, do_sample, num_beams,
                    temperature=(temperature if do_sample else None),
                    top_p=(top_p if do_sample else None),
                ))

            if getattr(self.dev, "type", "") == "cuda":
                torch.cuda.synchronize()

            out_text = " ".join(outs).strip()

            return {
                "task": "translate",
                "device": str(self.dev),
                "model": self.model_name,
                "params": {
                    "source_lang": src, "target_lang": tgt,
                    "max_new_tokens": max_new,
                    "do_sample": do_sample, "num_beams": num_beams,
                    "temperature": (temperature if do_sample else None),
                    "top_p": (top_p if do_sample else None),
                    "chunking": use_chunking, "chunk_len": chunk_len,
                },
                "input_chars": len(text),
                "output": out_text,
                "elapsed_sec": round(time.time() - t0, 3),
            }

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                if getattr(self.dev, "type", "") == "cuda":
                    torch.cuda.empty_cache()
                return {
                    "task": "translate",
                    "error": "CUDA OOM — جرّب تقليل max_new_tokens أو فعّل chunking أو شغّل على CPU"
                }
            return {"task": "translate", "error": str(e)}
        except Exception as e:
            return {"task": "translate", "error": str(e), "traceback": traceback.format_exc()}

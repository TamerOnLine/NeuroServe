from __future__ import annotations
import time
import traceback
from typing import List, Tuple, Optional, Dict

import torch
from transformers import (
    AutoTokenizer,
    MT5ForConditionalGeneration,
    MarianMTModel,
    MarianTokenizer,
)

from app.plugins.base import AIPlugin
from app.runtime import pick_device, pick_dtype

# خرائط أسماء اللغات الشائعة
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
            cur, cur_len = [w], len(w)
        else:
            cur.append(w)
            cur_len += wlen
    if cur:
        chunks.append(" ".join(cur))
    return chunks

# أزواج اللغات التي نستخدم لها Marian مباشرة (موصى به للترجمة الحقيقية)
_MARIAN_MAP: Dict[Tuple[str, str], str] = {
    ("arabic", "english"): "Helsinki-NLP/opus-mt-ar-en",
    ("english", "arabic"): "Helsinki-NLP/opus-mt-en-ar",
}

class Plugin(AIPlugin):
    tasks = ["translate"]

    def load(self) -> None:
        self.dev = pick_device()
        self.dtype = pick_dtype(str(self.dev))

        # mT5 (fallback)
        self.mt5_name = "google/mt5-small"
        self.mt5_tokenizer = AutoTokenizer.from_pretrained(self.mt5_name, use_fast=False)
        if self.mt5_tokenizer.pad_token_id is None and self.mt5_tokenizer.eos_token_id is not None:
            self.mt5_tokenizer.pad_token = self.mt5_tokenizer.eos_token

        self.mt5_model = MT5ForConditionalGeneration.from_pretrained(
            self.mt5_name,
            low_cpu_mem_usage=True,
            dtype=self.dtype
        ).to(self.dev).eval()

        # Marian نماذج تُحمَّل lazy عند أول استخدام لكل زوج
        self._marian_cache: Dict[str, Tuple[MarianTokenizer, MarianMTModel]] = {}

        # warmup خفيف
        try:
            _ = self._translate_mt5_once("hello", "english", "arabic",
                                         max_new=16, do_sample=False, num_beams=4)
        except Exception as e:
            print("[plugin][mt5] warmup warn:", e)

        print("[plugin] mt5 (with Marian fallback) loaded on", self.dev)

    # ---------------------- Marian backend ----------------------
    def _get_marian(self, model_name: str) -> Tuple[MarianTokenizer, MarianMTModel]:
        if model_name in self._marian_cache:
            return self._marian_cache[model_name]
        tok = MarianTokenizer.from_pretrained(model_name)
        mdl = MarianMTModel.from_pretrained(model_name, dtype=self.dtype).to(self.dev).eval()
        self._marian_cache[model_name] = (tok, mdl)
        return tok, mdl

    def _translate_marian_once(
        self,
        text: str,
        model_name: str,
        max_new: int,
        num_beams: int,
        do_sample: bool,
        temperature: Optional[float],
        top_p: Optional[float],
    ) -> str:
        tok, mdl = self._get_marian(model_name)
        enc = tok([text], return_tensors="pt", padding=True, truncation=True, max_length=512, return_attention_mask=True)
        enc = {k: v.to(self.dev) for k, v in enc.items()}
        gen_kwargs = dict(
            max_new_tokens=max_new,
            num_beams=(1 if do_sample else max(1, num_beams)),
            do_sample=do_sample,
            early_stopping=True,
        )
        if do_sample:
            if temperature is not None:
                gen_kwargs["temperature"] = float(temperature)
            if top_p is not None:
                gen_kwargs["top_p"] = float(top_p)

        with torch.no_grad():
            out_ids = mdl.generate(**enc, **gen_kwargs)
        return tok.batch_decode(out_ids, skip_special_tokens=True)[0].strip()

    # ---------------------- mT5 backend (fallback) ----------------------
    def _translate_mt5_once(
        self,
        text: str,
        src: str,
        tgt: str,
        max_new: int,
        do_sample: bool,
        num_beams: int,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        # صيغة T5: Capitalized lang names
        prefix = f"translate {src.capitalize()} to {tgt.capitalize()}: "
        enc = self.mt5_tokenizer(
            prefix + text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_attention_mask=True,
        )
        enc = {
            k: (
                v.to(self.dev, dtype=self.mt5_model.dtype) if (torch.is_tensor(v) and v.is_floating_point())
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
            pad_token_id=self.mt5_tokenizer.pad_token_id,
            eos_token_id=self.mt5_tokenizer.eos_token_id,
        )
        if do_sample:
            if temperature is not None:
                gen_kwargs["temperature"] = float(temperature)
            if top_p is not None:
                gen_kwargs["top_p"] = float(top_p)

        with torch.no_grad():
            out_ids = self.mt5_model.generate(**enc, **gen_kwargs)

        return self.mt5_tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()

    # ---------------------- Public API ----------------------
    def infer(self, payload: dict) -> dict:
        """
        يقبل:
        - text أو input: نص
        - source_lang / target_lang: رموز/أسماء شائعة (ar/en/...)
        خيارات:
        - max_new_tokens, do_sample, num_beams, temperature, top_p
        - chunking, chunk_len
        """
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

        # اختر backend
        marian_name = _MARIAN_MAP.get((src, tgt))
        backend = "marian" if marian_name else "mt5"

        try:
            if getattr(self.dev, "type", "") == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()

            outs: List[str] = []
            if backend == "marian":
                for p in parts:
                    outs.append(self._translate_marian_once(
                        p, marian_name, max_new, num_beams, do_sample,
                        (temperature if do_sample else None),
                        (top_p if do_sample else None),
                    ))
            else:
                for p in parts:
                    outs.append(self._translate_mt5_once(
                        p, src, tgt, max_new, do_sample, num_beams,
                        (temperature if do_sample else None),
                        (top_p if do_sample else None),
                    ))

            if getattr(self.dev, "type", "") == "cuda":
                torch.cuda.synchronize()

            out_text = " ".join(outs).strip()

            return {
                "task": "translate",
                "device": str(self.dev),
                "model": (marian_name or self.mt5_name),
                "backend": backend,
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
                    "error": "CUDA OOM — قلّل max_new_tokens أو فعّل chunking أو شغّل على CPU"
                }
            return {"task": "translate", "error": str(e)}
        except Exception as e:
            return {"task": "translate", "error": str(e), "traceback": traceback.format_exc()}

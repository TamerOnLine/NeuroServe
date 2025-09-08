from __future__ import annotations
import time, torch, traceback
from transformers import AutoTokenizer, MT5ForConditionalGeneration
from app.plugins.base import AIPlugin
from app.runtime import pick_device

_LANG_ALIASES = {
    # أسامي شائعة ↔ الصياغة الإنجليزية التي يفهمها mT5 prefix
    "ar": "arabic", "arabic": "arabic", "arabi": "arabic", "عربي": "arabic", "العربية": "arabic",
    "en": "english", "english": "english", "inglizi": "english", "انجليزي": "english", "الانجليزية": "english",
    "fr": "french", "français": "french", "فرنسي": "french",
    "de": "german", "deutsch": "german", "الالمانية": "german",
    "tr": "turkish", "turkish": "turkish", "تركي": "turkish",
    "es": "spanish", "spanish": "spanish", "اسباني": "spanish"
}
def _norm_lang(name: str, default: str) -> str:
    if not name: return default
    key = str(name).strip().lower()
    return _LANG_ALIASES.get(key, key)

class Plugin(AIPlugin):
    tasks = ["translate"]

    def load(self) -> None:
        self.dev = pick_device()
        self.model_name = "google/mt5-small"   # خفيف ومتعدد اللغات
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # حمّل على CPU ثم انقل للجهاز؛ استخدم dtype مناسب
        self.model = MT5ForConditionalGeneration.from_pretrained(
            self.model_name,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16 if self.dev.type == "cuda" else torch.float32
        ).to(self.dev)

        # pad_token احتياط
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # warmup خفيف
        try:
            _ = self._translate_once("hello", "english", "arabic", 16, False, 4)
        except Exception as e:
            print("[plugin][mt5] warmup warn:", e)

        print("[plugin] mt5 loaded on", self.dev)

    def _translate_once(self, text: str, src: str, tgt: str, max_new: int, do_sample: bool, num_beams: int) -> str:
        # صياغة mT5 القياسية: "translate <src> to <tgt>: <text>"
        prefix = f"translate {src} to {tgt}: "
        enc = self.tokenizer(prefix + text, return_tensors="pt", truncation=True, max_length=512).to(self.dev)
        with torch.no_grad():
            out_ids = self.model.generate(
                **enc,
                max_new_tokens=max_new,
                do_sample=do_sample,
                num_beams=(1 if do_sample else max(1, num_beams)),
                length_penalty=1.0,
                early_stopping=True,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(out_ids[0], skip_special_tokens=True)

    def infer(self, payload: dict) -> dict:
        # يقبل text أو input
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

        try:
            if self.dev.type == "cuda": torch.cuda.synchronize()
            t0 = time.time()
            out_text = self._translate_once(text, src, tgt, max_new, do_sample, num_beams)
            if self.dev.type == "cuda": torch.cuda.synchronize()
            return {
                "task": "translate",
                "device": str(self.dev),
                "model": self.model_name,
                "params": {"source_lang": src, "target_lang": tgt, "max_new_tokens": max_new, "do_sample": do_sample, "num_beams": num_beams},
                "input_chars": len(text),
                "output": out_text,
                "elapsed_sec": round(time.time() - t0, 3)
            }
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                if self.dev.type == "cuda": torch.cuda.empty_cache()
                return {"task": "translate", "error": "CUDA OOM — جرّب تقليل max_new_tokens أو شغّل على CPU"}
            return {"task":"translate","error":str(e)}
        except Exception as e:
            return {"task":"translate","error":str(e),"traceback":traceback.format_exc()}

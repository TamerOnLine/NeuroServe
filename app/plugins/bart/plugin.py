from __future__ import annotations
import time, torch, traceback
from transformers import AutoTokenizer, BartForConditionalGeneration
from app.plugins.base import AIPlugin
from app.runtime import pick_device, pick_dtype

class Plugin(AIPlugin):
    tasks = ["summarize"]

    def load(self) -> None:
        self.dev = pick_device()
        self.model_name = "facebook/bart-large-cnn"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # احتياط: تأكد من وجود pad_token
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # حمّل على CPU ثم انقل للجهاز
        self.model = BartForConditionalGeneration.from_pretrained(
            self.model_name,
            low_cpu_mem_usage=True,
            dtype=pick_dtype(str(self.dev))   # استبدال torch_dtype بـ dtype
        ).to(self.dev)

        # warmup خفيف لتسريع أول نداء
        try:
            _ = self._generate("Hello world.", max_length=30, min_length=5, do_sample=False)
        except Exception as e:
            print("[plugin][bart] warmup warn:", e)
        print("[plugin] bart loaded on", self.dev)

    def _generate(self, text: str, max_length: int, min_length: int, do_sample: bool):
        enc = self.tokenizer(
            [text],
            return_tensors="pt",
            truncation=True,
            max_length=1024,
            return_attention_mask=True,   # لإزالة تحذير attention mask
            padding=False
        ).to(self.dev)

        with torch.no_grad():
            out_ids = self.model.generate(
                **enc,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                num_beams=4 if not do_sample else 1,
                length_penalty=2.0,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        return self.tokenizer.decode(out_ids[0], skip_special_tokens=True)

    def infer(self, payload: dict) -> dict:
        # يقبل text أو input
        text = (payload.get("text") or payload.get("input") or "").strip()
        if not text:
            return {"task": "summarize", "error": "text is required"}

        max_length = int(payload.get("max_length", 180))
        min_length = int(payload.get("min_length", 60))
        do_sample = bool(payload.get("do_sample", False))

        # حواجز آمنة
        max_length = max(16, min(max_length, 512))
        min_length = max(5, min(min_length, max_length - 5))

        try:
            if self.dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()

            summary = self._generate(text, max_length, min_length, do_sample)

            if self.dev.type == "cuda":
                torch.cuda.synchronize()

            # تقدير الطول بشكل متوافق دائمًا (بدون return_length)
            enc_ids = self.tokenizer(
                text,
                add_special_tokens=True,
                truncation=False
            )["input_ids"]
            truncated = len(enc_ids) > 1024

            return {
                "task": "summarize",
                "device": str(self.dev),
                "model": self.model_name,
                "input_chars": len(text),
                "truncated_to_1024_tokens": bool(truncated),
                "params": {
                    "max_length": max_length,
                    "min_length": min_length,
                    "do_sample": do_sample
                },
                "summary": summary,
                "elapsed_sec": round(time.time() - t0, 3)
            }

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                if self.dev.type == "cuda":
                    torch.cuda.empty_cache()
                return {
                    "task": "summarize",
                    "error": "CUDA OOM — جرّب تقليل max_length أو شغّل على CPU"
                }
            return {"task": "summarize", "error": str(e)}
        except Exception as e:
            return {
                "task": "summarize",
                "error": str(e),
                "traceback": traceback.format_exc()
            }

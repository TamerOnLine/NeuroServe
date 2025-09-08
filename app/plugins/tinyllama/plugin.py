# app/plugins/tinyllama/plugin.py
from __future__ import annotations
import time, torch, traceback
from transformers import AutoModelForCausalLM, AutoTokenizer
from app.plugins.base import AIPlugin
from app.runtime import pick_device

class Plugin(AIPlugin):
    tasks = ["text-generation"]

    def load(self) -> None:
        self.dev = pick_device()
        self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            low_cpu_mem_usage=True,
            dtype=torch.float16 if self.dev.type == "cuda" else torch.float32,
        ).to(self.dev)

        # pad_token احتياطيًا لو غايب
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 🟢 warmup صغير لمنع 500 أول نداء
        try:
            _ = self.model.generate(
                **self.tokenizer("hello", return_tensors="pt").to(self.dev),
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        except Exception as e:
            print("[plugin][tinyllama] warmup warn:", e)

        print("[plugin] tinyllama loaded on", self.dev)

    def infer(self, payload: dict) -> dict:
        # يقبل input / prompt
        prompt = (payload.get("prompt") or payload.get("input") or "").strip()
        if not prompt:
            return {"task": "text-generation", "error": "prompt is required"}


        max_new = int(payload.get("max_new_tokens", 64))
        max_new = max(1, min(max_new, 256))  # حد علوي معقول
        temperature = float(payload.get("temperature", 0.7))
        top_p = float(payload.get("top_p", 0.9))

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.dev)

        # تنفيذ محمي ضد OOM وأخطاء أخرى
        try:
            if self.dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                out_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            if self.dev.type == "cuda":
                torch.cuda.synchronize()

            prompt_len = inputs["input_ids"].shape[1]
            gen_ids = out_ids[0][prompt_len:]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)

            return {
                "task": "text-generation",
                "device": str(self.dev),
                "model": self.model_name,
                "input": {"max_new_tokens": max_new, "temperature": temperature, "top_p": top_p},
                "output": text,
                "usage": {
                    "prompt_tokens": int(prompt_len),
                    "completion_tokens": int(gen_ids.shape[0]),
                    "total_tokens": int(prompt_len + gen_ids.shape[0]),
                },
                "elapsed_sec": round(time.time() - t0, 3),
            }

        except RuntimeError as e:
            # معالجة OOM
            if "CUDA out of memory" in str(e):
                if self.dev.type == "cuda":
                    torch.cuda.empty_cache()
                return {"task":"text-generation","error":"CUDA OOM — قلّل max_new_tokens أو جرّب على CPU"}
            return {"task":"text-generation","error":str(e)}

        except Exception as e:
            return {"task":"text-generation","error":str(e),"traceback":traceback.format_exc()}

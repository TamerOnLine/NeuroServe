from __future__ import annotations
import time, traceback, torch
from typing import Dict
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from app.plugins.base import AIPlugin
from app.runtime import pick_device, pick_dtype

# خرائط أكواد لغات M2M100
# وثّق فقط الأكثر شيوعًا، ويمكنك إضافة المزيد لاحقًا
_M2M_LANG = {
    "arabic":"ar", "ar":"ar", "العربية":"ar",
    "english":"en", "en":"en", "الانجليزية":"en",
    "french":"fr", "fr":"fr", "فرنسي":"fr",
    "german":"de", "de":"de",
    "spanish":"es", "es":"es",
    "turkish":"tr", "tr":"tr"
}
def _code(x:str, default:str)->str:
    key = (x or "").strip().lower()
    return _M2M_LANG.get(key, default)

class Plugin(AIPlugin):
    tasks = ["translate"]

    def load(self) -> None:
        self.dev = pick_device()
        self.dtype = pick_dtype(str(self.dev))
        self.name = "facebook/m2m100_418M"
        self.tokenizer = M2M100Tokenizer.from_pretrained(self.name)
        self.model = M2M100ForConditionalGeneration.from_pretrained(self.name, dtype=self.dtype).to(self.dev).eval()
        print("[plugin] translator_m2m loaded on", self.dev)

    def infer(self, payload:dict) -> dict:
        text = (payload.get("text") or payload.get("input") or "").strip()
        if not text:
            return {"task":"translate","error":"text is required"}

        src = _code(payload.get("source_lang","arabic"), "ar")
        tgt = _code(payload.get("target_lang","english"), "en")
        if src == tgt:
            return {"task":"translate","error":"source_lang and target_lang must be different"}

        max_new = max(8, min(int(payload.get("max_new_tokens", 64)), 256))
        num_beams = int(payload.get("num_beams", 4))
        do_sample = bool(payload.get("do_sample", False))

        try:
            self.tokenizer.src_lang = src
            enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, return_attention_mask=True)
            enc = {k: v.to(self.dev) for k, v in enc.items()}

            forced_bos = self.tokenizer.get_lang_id(tgt)

            if getattr(self.dev,"type","")=="cuda": torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                out_ids = self.model.generate(
                    **enc,
                    forced_bos_token_id=forced_bos,
                    max_new_tokens=max_new,
                    num_beams=(1 if do_sample else max(1, num_beams)),
                    do_sample=do_sample,
                    early_stopping=True
                )
            if getattr(self.dev,"type","")=="cuda": torch.cuda.synchronize()
            text_out = self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0].strip()

            return {
                "task":"translate","provider":"translator_m2m","device":str(self.dev),
                "model":self.name,"backend":"m2m100",
                "params":{"source_lang":src,"target_lang":tgt,"max_new_tokens":max_new,
                          "do_sample":do_sample,"num_beams":num_beams},
                "input_chars":len(text),"output":text_out,"elapsed_sec":round(time.time()-t0,3)
            }
        except Exception as e:
            return {"task":"translate","error":str(e),"traceback":traceback.format_exc()}

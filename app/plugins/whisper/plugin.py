from __future__ import annotations
import time, torch, torchaudio, traceback
from pathlib import Path
from urllib.parse import urlparse

import requests
from transformers import AutoProcessor, WhisperForConditionalGeneration
from app.plugins.base import AIPlugin
from app.runtime import pick_device



def infer(self, payload: dict) -> dict:
    # يقبل audio_url أو input
    audio_ref = payload.get("audio_url") or payload.get("input")
    if not audio_ref:
        return {"task": "speech-to-text", "error": "audio_url is required"}

    lang = payload.get("language")  # optional
    max_new = int(payload.get("max_new_tokens", 448))

    tmp_path = Path("tmp_audio.wav")

    try:
        # 🟢 تحقق إذا كان رابط
        parsed = urlparse(audio_ref)
        if parsed.scheme in ["http", "https"]:
            # نزّل الملف من الإنترنت
            self._download_audio(audio_ref, tmp_path)
            audio_path = tmp_path
        else:
            # استخدمه كمسار محلي مباشرة
            audio_path = Path(audio_ref)
            if not audio_path.exists():
                return {"task": "speech-to-text", "error": f"Local file not found: {audio_ref}"}

        # حمّل wav + resample إلى 16kHz
        waveform, sr = torchaudio.load(str(audio_path))
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)

        inputs = self.processor(waveform.squeeze(0), sampling_rate=16000, return_tensors="pt").to(self.dev)

        if self.dev.type == "cuda": torch.cuda.synchronize()
        t0 = time.time()

        forced_ids = None
        if lang:
            forced_ids = self.processor.get_decoder_prompt_ids(language=lang, task="transcribe")

        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new,
                forced_decoder_ids=forced_ids
            )

        if self.dev.type == "cuda": torch.cuda.synchronize()
        elapsed = round(time.time() - t0, 3)

        text = self.processor.batch_decode(out_ids, skip_special_tokens=True)[0]

        return {
            "task": "speech-to-text",
            "device": str(self.dev),
            "model": self.model_name,
            "language": lang or "auto",
            "audio_ref": str(audio_ref),
            "text": text,
            "elapsed_sec": elapsed
        }

    except Exception as e:
        return {"task":"speech-to-text","error":str(e),"traceback":traceback.format_exc()}
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

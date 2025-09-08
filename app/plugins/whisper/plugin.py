from __future__ import annotations
import time
from pathlib import Path
from urllib.parse import urlparse
import traceback
import requests
import numpy as np
import torch
import torchaudio

# ملاحظة: set_audio_backend متوقّفة في torchaudio → لا داعي لاستدعائها

from transformers import AutoProcessor, WhisperForConditionalGeneration
from app.plugins.base import AIPlugin
from app.runtime import pick_device, pick_dtype


class Plugin(AIPlugin):
    tasks = ["speech-to-text"]

    def load(self) -> None:
        self.dev = pick_device()
        self.model_name = "openai/whisper-small"  # غيّر الحجم حسب حاجتك
        self.processor = AutoProcessor.from_pretrained(self.model_name)

        # استخدم dtype الحديث واختره تلقائيًا حسب الجهاز
        dtype = pick_dtype(str(self.dev))
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_name,
            low_cpu_mem_usage=True,
            dtype=dtype,
        ).to(self.dev).eval()

        # warmup خفيف مع توحيد dtype للمدخلات العائمة فقط
        try:
            dummy = np.zeros(16000, dtype=np.float32)  # 1s صامت @16kHz
            inputs = self.processor(audio=dummy, sampling_rate=16000, return_tensors="pt")
            inputs = {
                k: (
                    v.to(self.dev, dtype=self.model.dtype) if (torch.is_tensor(v) and v.is_floating_point())
                    else (v.to(self.dev) if torch.is_tensor(v) else v)
                )
                for k, v in inputs.items()
            }
            _ = self.model.generate(**inputs, max_new_tokens=1)
        except Exception as e:
            print("[plugin][whisper] warmup warn:", e)

        print("[plugin] whisper loaded on", self.dev)

    def _download_audio(self, url: str, dst: Path, timeout: int = 30) -> None:
        r = requests.get(url, stream=True, timeout=timeout)
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    def _load_audio_16k_mono(self, path: Path):
        """
        يرجع waveform 1D @16kHz float32.
        يحاول torchaudio أولاً، ثم soundfile، ثم wave (WAV فقط) كخيار أخير.
        """
        # 1) torchaudio (الأفضل إن اشتغل)
        try:
            waveform, sr = torchaudio.load(str(path))  # [C, N]
        except Exception:
            # 2) soundfile (لو مثبت)
            try:
                import soundfile as sf
                data, sr = sf.read(str(path), dtype="float32")  # [N] أو [N, C]
                if data.ndim == 1:
                    waveform = torch.from_numpy(data).unsqueeze(0)  # [1, N]
                else:
                    waveform = torch.from_numpy(data).permute(1, 0)  # [C, N]
            except Exception:
                # 3) wave (WAV PCM فقط)
                import wave
                with wave.open(str(path), "rb") as wf:
                    sr = wf.getframerate()
                    n = wf.getnframes()
                    ch = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    raw = wf.readframes(n)
                # دعم 16-bit فقط هنا (للتبسيط)
                if sampwidth != 2:
                    raise RuntimeError("Unsupported WAV sample width; please use 16-bit PCM.")
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                if ch > 1:
                    audio = audio.reshape(-1, ch).mean(axis=1)
                waveform = torch.from_numpy(audio).unsqueeze(0)  # [1, N]

        # ستيريو→مونو
        if waveform.dim() == 2 and waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # إلى 16kHz
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
            sr = 16000

        return waveform.squeeze(0).contiguous(), sr

    def infer(self, payload: dict) -> dict:
        audio_ref = payload.get("audio_url") or payload.get("input")
        if not audio_ref:
            return {"task": "speech-to-text", "error": "audio_url (or input) is required"}

        # لو النص مش مسار/رابط، أعط رسالة واضحة
        if isinstance(audio_ref, str):
            p = Path(audio_ref)
            parsed = urlparse(audio_ref)
            if not p.exists() and parsed.scheme not in ("http", "https"):
                return {
                    "task": "speech-to-text",
                    "error": f"Expected audio file path or URL, got plain text: {audio_ref}"
                }

        # خيارات اختيارية
        lang = payload.get("language")  # مثال: "ar", "en", ...
        max_new = int(payload.get("max_new_tokens", 256))
        max_new = max(8, min(max_new, 1024))  # سقف منطقي
        tmp_path = Path("tmp_audio.wav")

        try:
            # تحميل من رابط أو مسار محلي
            parsed = urlparse(str(audio_ref))
            if parsed.scheme in ["http", "https"]:
                self._download_audio(str(audio_ref), tmp_path)
                audio_path = tmp_path
            else:
                audio_path = Path(str(audio_ref))
                if not audio_path.exists():
                    return {"task": "speech-to-text", "error": f"Local file not found: {audio_ref}"}

            # قراءة الصوت → 16kHz مونو
            mono, sr = self._load_audio_16k_mono(audio_path)

            # تجهيز المدخلات وتوحيد dtype مع الموديل (فقط للتنسورات العائمة)
            inputs = self.processor(audio=mono.numpy(), sampling_rate=sr, return_tensors="pt")
            inputs = {
                k: (
                    v.to(self.dev, dtype=self.model.dtype) if (torch.is_tensor(v) and v.is_floating_point())
                    else (v.to(self.dev) if torch.is_tensor(v) else v)
                )
                for k, v in inputs.items()
            }

            # ضبط max_new_tokens لتفادي تجاوز max_target_positions
            limit = getattr(self.model.config, "max_target_positions", 448)
            # تقدير طول البادئة
            if lang:
                try:
                    prompt_ids = self.processor.get_decoder_prompt_ids(language=lang, task="transcribe")
                    prompt_len = len(prompt_ids) if prompt_ids is not None else 0
                except Exception:
                    prompt_len = 0
            else:
                prompt_len = 2  # تقدير محافظ
            allowed_new = max(1, limit - prompt_len)
            if max_new > allowed_new:
                max_new = allowed_new

            # قياس الزمن
            if getattr(self.dev, "type", "") == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()

            # استخدم language/task بدل forced_decoder_ids (المهجّنة)
            gen_kwargs = {"max_new_tokens": max_new}
            if lang:
                gen_kwargs.update({"language": lang, "task": "transcribe"})

            with torch.no_grad():
                out_ids = self.model.generate(**inputs, **gen_kwargs)

            if getattr(self.dev, "type", "") == "cuda":
                torch.cuda.synchronize()
            elapsed = round(time.time() - t0, 3)

            text = self.processor.batch_decode(out_ids, skip_special_tokens=True)[0]

            return {
                "task": "speech-to-text",
                "device": str(self.dev),
                "model": self.model_name,
                "language": lang or "auto",
                "audio_ref": str(audio_ref),
                "text": text,
                "elapsed_sec": elapsed,
            }

        except Exception as e:
            return {"task": "speech-to-text", "error": str(e), "traceback": traceback.format_exc()}
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

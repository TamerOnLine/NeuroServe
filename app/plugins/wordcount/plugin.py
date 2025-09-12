from __future__ import annotations

from app.plugins.base import AIPlugin


class Plugin(AIPlugin):
    tasks = ["count"]

    def load(self) -> None:
        # ما في شي نحمّلو، مجرد طباعة
        print("[plugin] wordcount ready")

    def infer(self, payload: dict) -> dict:
        text = (payload.get("text") or payload.get("input") or "").strip()
        if not text:
            return {"task": "count", "error": "text is required"}

        words = len(text.split())
        return {"task": "count", "words": words, "input_chars": len(text)}

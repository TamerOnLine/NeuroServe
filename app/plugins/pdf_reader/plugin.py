from app.plugins.base import AIPlugin
import os
from pathlib import Path
from PyPDF2 import PdfReader

class Plugin(AIPlugin):
    tasks = ["extract-text"]

    def load(self) -> None:
        print("[plugin] pdf_reader ready")

    def _extract_text(self, file_path: str, max_pages: int = 5) -> str:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        reader = PdfReader(str(path))
        text = ""
        for i, page in enumerate(reader.pages):
            if i >= max_pages:
                break
            text += page.extract_text() + "\n"
        return text.strip()

    def infer(self, payload: dict) -> dict:
        file_path = payload.get("file_path")
        if not file_path:
            return {"task": "extract-text", "error": "file_path is required"}

        max_pages = int(payload.get("max_pages", 5))

        try:
            text = self._extract_text(file_path, max_pages=max_pages)
            return {
                "task": "extract-text",
                "file_path": file_path,
                "max_pages": max_pages,
                "output": text[:1000] + ("..." if len(text) > 1000 else ""),
                "chars_extracted": len(text)
            }
        except Exception as e:
            return {"task": "extract-text", "error": str(e)}

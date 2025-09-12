from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class AIPlugin(ABC):
    """واجهة قياسية لأي نموذج."""

    name: str = "unknown"  # اسم المزوّد (يُطابق اسم المجلد عادة)
    tasks: list[str] = []  # مثل: ["infer", "embed", "classify-image"]

    @abstractmethod
    def load(self) -> None:
        """تحميل النموذج والموارد في الذاكرة (مرة واحدة)."""
        ...

    @abstractmethod
    def infer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """نقطة الاستدلال العامة. يقرر الـ plugin كيف يفسّر payload."""
        ...

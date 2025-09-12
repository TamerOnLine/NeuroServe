import inspect
import time
from typing import Any, Dict, List, Optional

try:
    from app.plugins.base import AIPlugin
except Exception:

    class AIPlugin:
        tasks: List[str] = []

        def load(self, *_, **__): ...
        async def infer(self, payload: Dict[str, Any]): ...


# ✅ نستخدم الريجستري العالمي من loader.py كـ fallback
from app.plugins.loader import all_meta, get as get_plugin


def _deep_get(obj: Any, path: str, default: Any = "") -> Any:
    if not isinstance(path, str) or not path:
        return default
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def _subst(obj: Any, last: Any) -> Any:
    """
    استبدال المتغيّرات البسيطة المعتمدة على نتيجة الخطوة السابقة:
      - {{last.output}}
      - {{last.text}}
      - {{last.summary}}
      - {{last}}
    """

    def rep(s: str) -> str:
        if not isinstance(s, str):
            return s
        out = s
        if "{{last.output}}" in out:
            out = out.replace("{{last.output}}", str(_deep_get(last, "output", "")))
        if "{{last.text}}" in out:
            out = out.replace("{{last.text}}", str(_deep_get(last, "text", "")))
        if "{{last.summary}}" in out:
            out = out.replace("{{last.summary}}", str(_deep_get(last, "summary", "")))
        if "{{last}}" in out:
            out = out.replace("{{last}}", "" if last is None else str(last))
        return out

    if isinstance(obj, dict):
        return {k: _subst(v, last) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_subst(v, last) for v in obj]
    if isinstance(obj, str):
        return rep(obj)
    return obj


def _subst_item(obj: Any, item: Any) -> Any:
    """
    استبدال {{item}} عند استخدام op=map
    """

    def rep(s: str) -> str:
        return s.replace("{{item}}", str(item)) if isinstance(s, str) else s

    if isinstance(obj, dict):
        return {k: _subst_item(v, item) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_subst_item(v, item) for v in obj]
    if isinstance(obj, str):
        return rep(obj)
    return obj


class Plugin(AIPlugin):
    tasks = ["run"]

    def __init__(self):
        super().__init__()
        self.registry: Optional[Dict[str, Any]] = None

    def load(self, context: Optional[Dict[str, Any]] = None):
        """
        نحاول التقاط الريجستري من:
          - context["registry"] أو context["plugins"]
          - self.app.registry (إن وُجد)
        وإلا، سنعتمد على loader.get(...) كـ fallback داخل _get_plugin.
        """
        if isinstance(context, dict):
            if "registry" in context and isinstance(context["registry"], dict):
                self.registry = context["registry"]
            elif "plugins" in context and isinstance(context["plugins"], dict):
                self.registry = context["plugins"]
        if self.registry is None and hasattr(self, "app") and hasattr(self.app, "registry"):
            self.registry = getattr(self.app, "registry")

    def _get_plugin(self, provider: str):
        """
        ترتيب البحث:
          1) self.registry (إن كانت موجودة)
          2) الريجستري العالمي عبر loader.get(provider)
          3) خصائص إضافية app/manager/loader إن كانت تضع registry أو plugins
        """
        if isinstance(self.registry, dict) and provider in self.registry:
            return self.registry[provider]

        # ✅ fallback إلى الريجستري العالمي الذي عبّأته discover() عند بدء السيرفر
        p = get_plugin(provider)
        if p is not None:
            return p

        # محاولات إضافية (لو في بيئات أخرى تعلق registry/plugins)
        if hasattr(self, "plugins") and isinstance(self.plugins, dict) and provider in self.plugins:
            return self.plugins[provider]
        for attr in ("app", "manager", "loader"):
            scope = getattr(self, attr, None)
            if hasattr(scope, "registry") and isinstance(scope.registry, dict) and provider in scope.registry:
                return scope.registry[provider]
            if hasattr(scope, "plugins") and isinstance(scope.plugins, dict) and provider in scope.plugins:
                return scope.plugins[provider]
        return None

    async def _call_step(self, step_payload: Dict[str, Any]) -> Any:
        prov = step_payload.get("provider")
        task = step_payload.get("task")
        if not prov or not task:
            return {"error": "provider/task missing", "step": step_payload}
        plugin = self._get_plugin(prov)
        if plugin is None:
            return {"error": f"provider '{prov}' not found", "step": step_payload}
        out = plugin.infer(step_payload)
        if inspect.iscoroutine(out):
            out = await out
        return out

    async def _run(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        results = []
        last: Any = None
        timing_steps = []
        t_total0 = time.time()

        for idx, raw_step in enumerate(steps):
            name = f"step{idx + 1}"
            t0 = time.time()

            # دعم map: يكرّر خطوة واحدة على items
            if raw_step.get("op") == "map":
                items = raw_step.get("items", [])
                base = raw_step.get("step", {})
                mapped_out = []
                for it in items if isinstance(items, list) else []:
                    payload = _subst(base, last)
                    payload = _subst_item(payload, it)
                    o = await self._call_step(payload)
                    mapped_out.append(o)
                last = mapped_out
                timing_steps.append({"name": name, "op": "map", "elapsed_ms": round((time.time() - t0) * 1000.0, 3)})
                results.append({"name": name, "op": "map", "items_len": len(items), "out": mapped_out})
                continue

            # خطوة عادية
            payload = _subst(raw_step, last)
            out = await self._call_step(payload)
            last = out
            timing_steps.append({"name": name, "op": "infer", "elapsed_ms": round((time.time() - t0) * 1000.0, 3)})
            results.append({"name": name, "op": "infer", "step": raw_step, "out": out})

        total_ms = round((time.time() - t_total0) * 1000.0, 3)
        return {"steps": results, "timing": {"elapsed_ms": total_ms, "steps": timing_steps}, "last": last}

    async def infer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        steps = payload.get("steps")
        if not isinstance(steps, list) or not steps:
            return {"provider": "workflow", "task": "run", "error": "steps[] is required"}

        # (اختياري) طباعة Debug لأسماء المزودين المتاحين
        try:
            meta = all_meta()
            # يمكنك إزالة السطر التالي إن ما بدك ضجيج في اللوج:
            print("[workflow] available providers:", sorted(meta.keys()))
        except Exception:
            pass

        result = await self._run(steps)
        return {"provider": "workflow", "task": "run", **result}

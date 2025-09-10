from __future__ import annotations
import importlib.util, json, os, pathlib, traceback
from typing import Dict, Any, Tuple
from .base import AIPlugin

PLUGIN_DIR = pathlib.Path(__file__).resolve().parent

_registry: Dict[str, AIPlugin] = {}
_meta: Dict[str, Dict[str, Any]] = {}

def _load_manifest(folder: pathlib.Path) -> Dict[str, Any]:
    mf = folder / "manifest.json"
    if mf.exists():
        try:
            return json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def _load_module(folder: pathlib.Path):
    mod_path = folder / "plugin.py"
    if not mod_path.exists():
        return None
    spec = importlib.util.spec_from_file_location(f"plugins.{folder.name}", str(mod_path))
    if not spec or not spec.loader:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module

def discover(reload: bool = False) -> Tuple[Dict[str, AIPlugin], Dict[str, Dict[str, Any]]]:
    """Scan plugins/* and load each plug-in with its manifest."""
    global _registry, _meta
    if reload:
        _registry, _meta = {}, {}

    if not PLUGIN_DIR.exists():
        return _registry, _meta

    for folder in sorted(PLUGIN_DIR.iterdir()):
        if not folder.is_dir():
            continue
        name = folder.name

        # 🛑 Filter out non-plugin directories
        if name.startswith(".") or name == "__pycache__":
            continue

        if name in _registry:
            continue
        try:
            module = _load_module(folder)
            if not module or not hasattr(module, "Plugin"):
                continue
            plugin_cls = getattr(module, "Plugin")
            plugin: AIPlugin = plugin_cls()
            plugin.name = name
            plugin.load()
            _registry[name] = plugin
            _meta[name] = {"name": name, **_load_manifest(folder)}
        except Exception:
            print(f"[plugin] failed to load '{name}':\n{traceback.format_exc()}")
    return _registry, _meta

def get(name: str) -> AIPlugin | None:
    return _registry.get(name)

def all_meta() -> Dict[str, Dict[str, Any]]:
    return _meta

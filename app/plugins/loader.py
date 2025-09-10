from __future__ import annotations
import importlib.util, json, os, pathlib, traceback
from typing import Dict, Any, Tuple
from .base import AIPlugin

PLUGIN_DIR = pathlib.Path(__file__).resolve().parent

_registry: Dict[str, AIPlugin] = {}
_meta: Dict[str, Dict[str, Any]] = {}

def _load_manifest(folder: pathlib.Path) -> Dict[str, Any]:
    """Load plugin manifest.json if it exists."""
    mf = folder / "manifest.json"
    if mf.exists():
        try:
            return json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def _load_module(folder: pathlib.Path):
    """Load plugin.py as a Python module."""
    mod_path = folder / "plugin.py"
    if not mod_path.exists():
        return None
    spec = importlib.util.spec_from_file_location(f"plugins.{folder.name}", str(mod_path))
    if not spec or not spec.loader:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module

def _parse_csv_env(name: str) -> set[str]:
    """Helper to parse PLUGINS_ALLOW / PLUGINS_DENY env vars into a set."""
    raw = os.getenv(name, "") or ""
    return {x.strip() for x in raw.split(",") if x.strip()}

def discover(reload: bool = False) -> Tuple[Dict[str, AIPlugin], Dict[str, Dict[str, Any]]]:
    """
    Scan plugins/* and load plugins + manifests.

    Environment variables respected:
      - CI_LIGHT_MODE=1  -> only load lightweight plugins (skip heavy ones).
      - DISABLE_PLUGINS=1 -> skip loading all plugins (metadata only).
      - PLUGINS_ALLOW=comma,separated,names -> explicitly allow certain plugins.
      - PLUGINS_DENY=comma,separated,names  -> explicitly deny certain plugins.
    """
    global _registry, _meta
    if reload:
        _registry, _meta = {}, {}

    if not PLUGIN_DIR.exists():
        return _registry, _meta

    # Flags
    ci_light = os.getenv("CI_LIGHT_MODE", "0").lower() in ("1", "true", "yes")
    disable_all = os.getenv("DISABLE_PLUGINS", "0").lower() in ("1", "true", "yes")

    # Allow/Deny lists
    allow_from_env = _parse_csv_env("PLUGINS_ALLOW")
    deny_from_env = _parse_csv_env("PLUGINS_DENY")

    # Default safe allow-list in CI light mode
    default_ci_allow = {
        "dummy",
        "pdf_reader",
        "dichfoto_proxy",
        # Add other lightweight plugins if needed: "wordcount", "tinynet", ...
    }

    for folder in sorted(PLUGIN_DIR.iterdir()):
        if not folder.is_dir():
            continue
        name = folder.name
        if name in _registry:
            continue
        try:
            # Always read manifest (even if skipped)
            manifest = _load_manifest(folder)

            # 1) Global disable
            if disable_all:
                _meta[name] = {"name": name, **manifest, "skipped": "disabled_all"}
                continue

            # 2) Explicit deny
            if name in deny_from_env:
                _meta[name] = {"name": name, **manifest, "skipped": "denied"}
                continue

            # 3) CI light mode
            if ci_light:
                allowed = allow_from_env or default_ci_allow
                if name not in allowed:
                    _meta[name] = {"name": name, **manifest, "skipped": "ci_light_mode"}
                    continue

            module = _load_module(folder)
            if not module or not hasattr(module, "Plugin"):
                _meta[name] = {"name": name, **manifest, "skipped": "no_plugin_module"}
                continue

            plugin_cls = getattr(module, "Plugin")
            plugin: AIPlugin = plugin_cls()
            plugin.name = name

            # Heavy init happens here
            plugin.load()

            _registry[name] = plugin
            _meta[name] = {"name": name, **manifest}

        except Exception:
            print(f"[plugin] failed to load '{name}':\n{traceback.format_exc()}")
    return _registry, _meta

def get(name: str) -> AIPlugin | None:
    return _registry.get(name)

def all_meta() -> Dict[str, Dict[str, Any]]:
    return _meta

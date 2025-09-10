from __future__ import annotations
import psutil, torch, traceback
from typing import Dict, Any
from app.plugins.base import AIPlugin
from app.plugins.loader import discover, all_meta, get

class Plugin(AIPlugin):
    tasks = ["list"]

    def load(self) -> None:
        # No heavy loading needed
        print("[plugin] services monitor loaded ✅")

    def infer(self, payload: dict) -> dict:
        """
        Return info about all services (plugins).
        """
        try:
            # Re-scan plugins dynamically
            discover(reload=True)
            meta = all_meta()  # manifest data per plugin

            services = []
            for name, info in meta.items():
                # Skip if no plugin module exists
                plugin = get(name)
                if info.get("skipped") == "no_plugin_module":
                    continue

                # Metadata from manifest (with defaults)
                title   = info.get("title")   or info.get("name") or name
                ptype   = info.get("type")    or "unknown"
                ver     = info.get("version") or "N/A"
                author  = info.get("author")  or "N/A"
                license = info.get("license") or "N/A"
                tasks   = getattr(plugin, "tasks", []) if plugin else []

                # Device formatting
                dev = None
                if plugin and hasattr(plugin, "dev"):
                    d = plugin.dev
                    dev = f"{getattr(d, 'type', d)}"
                    if getattr(d, "type", None) == "cuda":
                        idx = getattr(d, "index", 0)
                        dev = f"cuda:{idx}"

                services.append({
                    "name": name,
                    "title": title,
                    "type":  ptype,
                    "status": "running" if plugin else "not_loaded",
                    "device": dev if plugin else None,
                    "loaded": plugin is not None,
                    "version": ver,
                    "author": author,
                    "license": license,
                    "tasks": tasks,
                    "resources": {
                        "cpu_percent": psutil.cpu_percent(),
                        "ram_mb": int(psutil.virtual_memory().used / 1e6),
                        "gpu_memory_mb": (
                            int(torch.cuda.memory_allocated(0) / 1e6)
                            if torch.cuda.is_available() else 0
                        )
                    },
                    "endpoint": f"/plugins/{name}/infer"
                })

            return {"task": "list", "services": services}

        except Exception as e:
            return {
                "task": "list",
                "error": str(e),
                "traceback": traceback.format_exc()
            }

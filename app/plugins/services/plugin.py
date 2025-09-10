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
            meta = all_meta()

            services = []
            for name, info in meta.items():
                plugin = get(name)
                services.append({
                    "name": name,
                    "title": info.get("title", name),
                    "type": info.get("type", "unknown"),
                    "status": "running" if plugin else "not_loaded",
                    "device": getattr(getattr(plugin, "dev", None), "type", "cpu") if plugin else None,
                    "loaded": plugin is not None,
                    "version": info.get("version", "N/A"),
                    "author": info.get("author", "N/A"),
                    "license": info.get("license", "N/A"),
                    "tasks": getattr(plugin, "tasks", []),
                    "resources": {
                        "cpu_percent": psutil.cpu_percent(),
                        "ram_mb": int(psutil.virtual_memory().used / 1e6),
                        "gpu_memory_mb": (
                            int(torch.cuda.memory_allocated(0) / 1e6) if torch.cuda.is_available() else 0
                        )
                    },
                    "endpoint": f"/plugins/{name}/infer"
                })

            return {
                "task": "list",
                "services": services
            }

        except Exception as e:
            return {
                "task": "list",
                "error": str(e),
                "traceback": traceback.format_exc()
            }

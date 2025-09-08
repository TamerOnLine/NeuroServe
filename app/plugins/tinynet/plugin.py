from __future__ import annotations
import torch, time
from app.plugins.base import AIPlugin
from app.runtime import pick_device
from app.toy_model import TinyNet

class Plugin(AIPlugin):
    tasks = ["infer"]

    def load(self) -> None:
        self.dev = pick_device()
        self.model = TinyNet().to(self.dev).eval()
        with torch.no_grad():
            _ = self.model(torch.randn(1, 512, device=self.dev))  # warm
        print("[plugin] tinynet loaded on", self.dev)

    def infer(self, payload: dict) -> dict:
        batch = int(payload.get("batch", 4))
        in_features = 512
        with torch.no_grad():
            x = torch.randn(batch, in_features, device=self.dev)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t0 = time.time()
            y = self.model(x)
            if torch.cuda.is_available(): torch.cuda.synchronize()
        return {
            "task": "infer",
            "device": str(self.dev),
            "out_shape": list(y.shape),
            "elapsed_sec": round(time.time() - t0, 4),
        }

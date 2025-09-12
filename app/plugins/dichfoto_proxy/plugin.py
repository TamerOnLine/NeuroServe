import requests

from app.plugins.base import AIPlugin


class Plugin(AIPlugin):
    tasks = ["forward"]

    def load(self) -> None:
        # لو السيرفر عنده دومين عام
        self.base_url = "https://dichfoto.com"
        print("[plugin] dichfoto_proxy ready (target:", self.base_url, ")")

    def infer(self, payload: dict) -> dict:
        endpoint = payload.get("endpoint", "/")
        method = payload.get("method", "GET").upper()
        data = payload.get("data", {})

        url = self.base_url + endpoint
        try:
            if method == "GET":
                resp = requests.get(url, params=data)
            else:
                resp = requests.post(url, json=data)

            return {
                "task": "forward",
                "endpoint": endpoint,
                "status_code": resp.status_code,
                "data": resp.json()
                if resp.headers.get("content-type", "").startswith("application/json")
                else resp.text,
            }
        except Exception as e:
            return {"task": "forward", "error": str(e), "endpoint": endpoint}

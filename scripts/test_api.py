# tests/test_api.py
import os
import requests
from fastapi.testclient import TestClient
from app.main import app

BASE_URL = os.getenv("BASE_URL")
_client = None if BASE_URL else TestClient(app)


def get_json(path: str, method: str = "get", **kwargs):
    """Helper to fetch JSON from API (works with BASE_URL or TestClient)."""
    if BASE_URL:
        resp = requests.request(method, f"{BASE_URL}{path}", timeout=10, **kwargs)
    else:
        resp = _client.request(method, path, **kwargs)
    assert resp.status_code == 200
    return resp.json()


def test_health():
    body = get_json("/health")
    assert isinstance(body, dict)
    assert body.get("ok", True)  # allow {"ok": true}


def test_cuda_info():
    body = get_json("/cuda_info")
    assert isinstance(body, dict)
    assert "device_count" in body or "cuda" in body


def test_matmul():
    body = get_json("/matmul", method="post", json={"n": 512})
    assert isinstance(body, dict)
    assert "elapsed_ms" in body


def test_infer():
    body = get_json("/infer", method="post", json={"text": "hello world"})
    assert isinstance(body, dict)
    assert "task" in body or "output" in body

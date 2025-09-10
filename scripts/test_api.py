import requests

BASE = "http://127.0.0.1:8000"

def safe_print_json(label, r):
    """حاول طباعة JSON، ولو فشل اطبع الرد الخام"""
    try:
        data = r.json()
        print(f"{label}:", data)
    except Exception:
        print(f"{label} (non-JSON): status={r.status_code}, body={r.text[:200]}")

def test_health():
    r = requests.get(f"{BASE}/health")
    safe_print_json("Health", r)

def test_cuda_info():
    r = requests.get(f"{BASE}/cuda_info")
    safe_print_json("CUDA info", r)

def test_matmul():
    r = requests.post(f"{BASE}/matmul", json={"n": 2048})
    safe_print_json("Matmul", r)

def test_infer():
    r = requests.post(f"{BASE}/infer", json={"text": "hello world"})
    safe_print_json("Infer", r)

if __name__ == "__main__":
    test_health()
    test_cuda_info()
    test_matmul()
    test_infer()

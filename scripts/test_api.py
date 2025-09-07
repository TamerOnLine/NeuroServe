import requests

BASE_URL = "http://localhost:8000"

def test_health():
    r = requests.get(f"{BASE_URL}/health")
    print("Health:", r.json())

def test_cuda():
    r = requests.get(f"{BASE_URL}/cuda")
    print("CUDA info:", r.json())

def test_matmul(n=2048):
    r = requests.post(f"{BASE_URL}/matmul", json={"n": n})
    print("Matmul:", r.json())

def test_infer(batch=16):
    r = requests.post(f"{BASE_URL}/infer", json={"batch": batch})
    print("Infer:", r.json())

if __name__ == "__main__":
    test_health()
    test_cuda()
    test_matmul()
    test_infer()

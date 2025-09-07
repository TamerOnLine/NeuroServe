# 🚀 gpu_server

A REST API server built with **FastAPI + PyTorch** for experiments on
**GPU / CPU**.\
It automatically runs computations on the GPU (CUDA) if available, or
falls back to CPU.

------------------------------------------------------------------------

## ✨ Features

-   ✅ Supports **GPU (CUDA)** or **CPU** with the same code.
-   ✅ API powered by **FastAPI** with auto docs at `/docs`.
-   ✅ Matrix multiplication benchmark endpoint.
-   ✅ TinyNet demo model for inference tests.
-   ✅ Configurable via `.env` file.

------------------------------------------------------------------------

## 📂 Project Structure

    gpu-server/
    ├─ app/
    │  ├─ __init__.py
    │  ├─ main.py
    │  ├─ runtime.py
    │  └─ toy_model.py
    ├─ scripts/
    │  ├─ install_torch_auto.py
    │  └─ test_api.py
    ├─ tests/
    │  └─ test_endpoints.py        
    ├─ requirements.txt            
    ├─ requirements.lock.txt       
    ├─ .env.example
    ├─ .gitignore
    └─ README.md


------------------------------------------------------------------------

## ⚙️ Requirements

-   Python **3.12+**
-   NVIDIA Driver (optional for GPU)
-   CUDA Toolkit (optional for development)

------------------------------------------------------------------------

## 🛠️ Installation

### 1) Create virtual environment

``` powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install base dependencies

``` powershell
pip install -r requirements.txt
```

### 3) Install correct PyTorch (GPU or CPU)

``` powershell
python install_torch_auto.py
```

> 🔹 The script checks your system and installs the right wheel: -
> `torch+cu124` if a compatible GPU is found. - `torch+cpu` if no GPU is
> available.

------------------------------------------------------------------------

## ⚡ Running the server

### 1) With Uvicorn CLI

``` powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2) With Python directly

``` powershell
python app/main.py
```

------------------------------------------------------------------------

## ⚙️ Environment Variables

Create a `.env` file in the project root:

``` ini
# Run on GPU (if available)
DEVICE=cuda:0

# Or force CPU mode
# DEVICE=cpu

# Matrix warmup size (optional)
WARMUP_MATMUL_SIZE=1024
```

------------------------------------------------------------------------

## 📡 API Endpoints

-   ✅ **GET** `/health` → Server health check\
-   ✅ **GET** `/cuda` → PyTorch + CUDA + device info\
-   ✅ **POST** `/matmul` → Run matrix multiplication on device\
-   ✅ **POST** `/infer` → Run TinyNet inference

------------------------------------------------------------------------

## 🔍 Testing

### With browser

Visit:

    http://localhost:8000/docs

to try all endpoints interactively.

### With Python (`test_api.py`)

``` python
import requests

BASE_URL = "http://localhost:8000"

print(requests.get(f"{BASE_URL}/health").json())
print(requests.get(f"{BASE_URL}/cuda").json())
print(requests.post(f"{BASE_URL}/matmul", json={"n": 2048}).json())
print(requests.post(f"{BASE_URL}/infer", json={"batch": 16}).json())
```

------------------------------------------------------------------------

## 📊 Example Output

``` json
{
  "torch_version": "2.6.0+cu124",
  "torch_cuda_version": "12.4",
  "cuda_available": true,
  "device": "cuda:0",
  "gpu_name": "NVIDIA GeForce RTX 3080",
  "total_memory_gb": 10.0
}
```

------------------------------------------------------------------------

## 🚀 Roadmap

-   [ ] Add support for real PyTorch models (ResNet, LLaMA, BERT).
-   [ ] File upload endpoint (images/text) for inference.
-   [ ] Docker + NVIDIA Container Toolkit support.

------------------------------------------------------------------------

## 📜 License

MIT © TamerOnLine

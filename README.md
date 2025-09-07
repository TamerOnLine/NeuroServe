# NeuroServe (GPU/CPU FastAPI Server)

A lightweight REST API server built with **FastAPI** and **PyTorch** that runs the same code on **GPU (CUDA)** when available, and **falls back to CPU** automatically. It ships with a tiny demo model, performance probes, and a simple web UI to exercise endpoints.

---

## Highlights
- ✅ Auto‑selects **GPU or CPU** at runtime (configurable via `.env`).
- ✅ Clean **FastAPI** endpoints with Swagger UI (`/docs`) and ReDoc (`/redoc`).
- ✅ Built‑in **control panel** (`/ui`) and a **model size calculator** (`/tools/model-size`).
- ✅ Example **TinyNet** model + inference & matmul benchmarks.
- ✅ Works on Windows/Linux/macOS (CPU), NVIDIA CUDA on Windows/Linux.

---

## Project Structure (key files)
```
app/
  main.py           # FastAPI app & endpoints
  runtime.py        # device picking, CUDA info, warmup
  toy_model.py      # TinyNet demo model
  templates/
    index.html      # quick links
    ui.html         # control panel
    model_size.html # model size calculator
scripts/
  install_torch.py  # cross‑platform PyTorch installer (GPU/CPU)
  test_api.py       # quick client to hit endpoints
requirements.txt
README.md
```

---

## Requirements
- **Python 3.12+** (recommended)
- **Windows or Linux** (GPU or CPU) / **macOS** (CPU/MPS)
- **NVIDIA driver** for CUDA (optional – only if you want GPU)

> Tip (Windows): If you ever need to build packages from source, install **Visual Studio 2022 Build Tools** (MSVC, CMake, Ninja). Not required for normal use when PyTorch wheels are available.

---

## Quickstart

### 1) Create a virtual environment
**Windows (PowerShell)**
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 2) Install base dependencies
```bash
pip install -r requirements.txt
```

### 3) Install the right PyTorch (GPU if available, otherwise CPU)
```bash
python -m scripts.install_torch
```
> The script auto‑detects your platform (NVIDIA, ROCm, macOS/CPU) and installs appropriate wheels.

### 4) Run the server
**Option A – Uvicorn**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
**Option B – Python**
```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open:
- Swagger: http://localhost:8000/docs
- Control Panel: http://localhost:8000/ui

---

## Configuration (.env)
Create a `.env` file in the project root to control device & warmup:
```ini
# Prefer the first CUDA GPU if available; otherwise CPU will be used
DEVICE=cuda:0

# Force CPU mode (uncomment to override)
# DEVICE=cpu

# Size used by warmup matmul (int)
WARMUP_MATMUL_SIZE=1024
```

---

## API Endpoints
| Method | Path              | Description                              |
|-------:|-------------------|------------------------------------------|
| GET    | `/health`         | Health check                             |
| GET    | `/cuda`           | PyTorch/CUDA/device info                 |
| GET    | `/env`            | Short environment summary (add `?pretty=1`) |
| GET    | `/env/full`       | Extended env + GPU list (add `?pretty=1`) |
| GET    | `/env/system`     | OS/CPU/RAM info (add `?pretty=1`)        |
| POST   | `/matmul`         | Matrix multiply benchmark `{n:int}`      |
| POST   | `/infer`          | TinyNet inference `{batch:int}`          |
| POST   | `/run/test-api`   | Runs `scripts/test_api.py` and returns stdout |
| GET    | `/`               | Home with quick links                    |
| GET    | `/ui`             | Control panel (interactive)              |
| GET    | `/tools/model-size` | MLP model size calculator               |

### Request examples
**Matrix multiply**
```bash
curl -X POST http://localhost:8000/matmul \\
  -H "Content-Type: application/json" \\
  -d '{"n": 2048}'
```

**Inference**
```bash
curl -X POST http://localhost:8000/infer \\
  -H "Content-Type: application/json" \\
  -d '{"batch": 16}'
```

**PowerShell (Invoke-RestMethod)**
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8000/matmul" -Method POST -Body '{"n": 1024}' -ContentType "application/json"
```

---

## Control Panel (UI)
Open **`/ui`** to:
- View current **device**, **CUDA**, and **Python** info.
- Run **matmul** and **inference** interactively.
- Trigger the **test API** client and read its output in the page.

There’s also a **Model Size Calculator** at **`/tools/model-size`** that estimates parameter count and memory footprint for a simple MLP and suggests whether CPU or GPU is recommended.

---

## Tiny client: `scripts/test_api.py`
A minimal Python client that calls all core endpoints. Run it while the server is up:
```bash
python -m scripts.test_api
```
Expected output includes JSON from `/health`, `/cuda`, `/matmul`, and `/infer`.

---

## Troubleshooting
- **Torch import error or missing wheels** → Use **Python 3.12** in your venv and rerun the installer: `python -m scripts.install_torch`.
- **No GPU detected** → The server will safely use CPU. To force CPU explicitly: `DEVICE=cpu` in `.env`.
- **Windows**: If you hit build errors (rare), install **VS 2022 Build Tools** and ensure you’re on official PyTorch wheels.
- **CUDA mismatch**: If `nvidia-smi` shows a much newer driver than your installed torch+cuXX, reinstall torch for a matching CUDA runtime.

---

## Roadmap
- [ ] Real models: ResNet/BERT/LLaMA loaders & sample inputs.
- [ ] File upload endpoints (images/text) with pre/post‑processing.
- [ ] Docker images (CPU & NVIDIA CUDA) + `docker-compose`.
- [ ] Pre‑commit hooks (ruff/black) & GitHub Actions CI.

---

## License
MIT © TamerOnLine

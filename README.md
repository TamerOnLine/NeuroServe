# 🧠 NeuroServe – GPU/CPU FastAPI AI Server

[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Framework-green)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI (CPU, 3 OS)](https://github.com/TamerOnLine/NeuroServe/actions/workflows/ci.yml/badge.svg)](https://github.com/TamerOnLine/NeuroServe/actions/workflows/ci.yml)
[![CI (GPU, self-hosted)](https://github.com/TamerOnLine/NeuroServe/actions/workflows/gpu-ci.yml/badge.svg)](https://github.com/TamerOnLine/NeuroServe/actions/workflows/gpu-ci.yml)


A lightweight **REST API server** powered by **FastAPI** & **PyTorch**.  
It runs seamlessly on **GPU (CUDA)** if available, and safely falls back to **CPU**.  

🚀 Includes a demo model, performance probes, a **web control panel**, and an extendable **Plugin System** for real AI models.

---

## ✨ Features
- ✅ Auto-selects **GPU or CPU** at runtime (`DEVICE` in `.env`).
- ✅ Clean **FastAPI** endpoints with Swagger UI (`/docs`) and ReDoc (`/redoc`).
- ✅ Interactive **Control Panel** (`/ui`) & **Model Size Calculator** (`/tools/model-size`).
- ✅ Example **TinyNet** model + benchmarks (`/matmul`, `/infer`).
- ✅ Extensible **Plugin System** with ready-to-use AI models.
- ✅ Works on **Windows/Linux/macOS** (CPU) & **CUDA** (NVIDIA GPUs).

---

## 📂 Project Structure
```
app/
  main.py           # FastAPI app & endpoints
  runtime.py        # device selection, CUDA info, warmup
  toy_model.py      # TinyNet demo model
  plugins/          # Modular plugins (bart, clip, resnet18, ner, etc.)
  templates/        # HTML templates for UI
scripts/
  install_torch.py  # Auto-installs correct PyTorch (CPU/GPU)
  test_api.py       # Quick client to test endpoints
requirements.txt
README.md
```

---

## ⚡ Quickstart

### 1. Setup Environment
**Windows (PowerShell)**:
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS**:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
python -m scripts.install_torch
```

### 3. Run the Server
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open in browser:
- **Swagger UI** → http://localhost:8000/docs  
- **Control Panel** → http://localhost:8000/ui  
- **Plugin Console** → http://localhost:8000/plugins/ui  
- **Infer Client** → http://localhost:8000/infer-client  

---

## ⚙️ Configuration (`.env`)
```ini
# Prefer the first CUDA GPU if available
DEVICE=cuda:0

# Force CPU mode (override)
# DEVICE=cpu

# Warmup matrix size (for benchmarks)
WARMUP_MATMUL_SIZE=1024
```

---

## 🔌 API Endpoints

| Method | Path                | Description                      |
|--------|---------------------|----------------------------------|
| GET    | `/health`           | Health check                     |
| GET    | `/cuda`             | CUDA / device info               |
| GET    | `/env`              | Environment summary              |
| GET    | `/env/full`         | Full environment + GPU list      |
| GET    | `/env/system`       | OS/CPU/RAM system info           |
| POST   | `/matmul`           | Matrix multiply benchmark        |
| POST   | `/infer`            | TinyNet inference                |
| POST   | `/inference`        | Generic plugin inference         |
| GET    | `/plugins`          | List available plugins           |
| GET    | `/ui`               | Interactive control panel        |
| GET    | `/tools/model-size` | Model size calculator (UI)       |

---

## 🧩 Plugins
NeuroServe uses a **modular plugin system** under `app/plugins/`.

Available plugins:
- **bart** → Text summarization (`facebook/bart-large-cnn`)
- **clip** → Text ↔ Image embeddings & similarity
- **distilbert** → Sentiment classification
- **resnet18** → Image classification (ImageNet)
- **ner** → Named Entity Recognition
- **tinyllama** → Lightweight text generation
- **translator / translator_m2m** → Translation
- **pdf_reader** → Extract text from PDFs
- **dummy** → Ping test
- **dichfoto_proxy** → Forwarding proxy

🔧 **Build your own Plugin (5 min)**
```python
from app.plugins.base import AIPlugin

class Plugin(AIPlugin):
    tasks = ["hello"]

    def load(self):
        print("[plugin] hello ready")

    def infer(self, payload):
        name = payload.get("name", "world")
        return {"task": "hello", "message": f"Hello, {name}!"}
```
Add it in `app/plugins/hello/` with `manifest.json`.

---

## 🖥️ Example Requests

**Matrix Multiply**
```bash
curl -X POST http://localhost:8000/matmul   -H "Content-Type: application/json"   -d '{"n": 2048}'
```

**BART Summarization**
```bash
curl -X POST http://localhost:8000/inference   -H "Content-Type: application/json"   -d '{"provider":"bart","task":"summarize","text":"Deep learning is a subfield..."}'
```

**NER Entity Extraction**
```bash
curl -X POST http://localhost:8000/inference   -H "Content-Type: application/json"   -d '{"provider":"ner","task":"extract-entities","text":"Barack Obama was born in Hawaii."}'
```

---

## 🛠️ Troubleshooting
- **Torch import error** → Ensure Python 3.12 + rerun `python -m scripts.install_torch`.
- **No GPU detected** → Falls back to CPU. Force with `DEVICE=cpu`.
- **CUDA mismatch** → Reinstall torch with matching CUDA runtime.
- **Out of Memory (OOM)** → Reduce `max_length` or use CPU mode.
- **Windows Execution Policy** →  
  ```powershell
  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
  ```

---

## 📍 Roadmap
- [ ] File upload endpoints (images/text).
- [ ] Docker images (CPU & GPU).
- [ ] Extended plugin demo UI.
- [ ] CI/CD with pre-commit hooks & GitHub Actions.
- [x] Real AI plugins (BART, CLIP, ResNet, DistilBERT, NER, etc.).

---

## 📜 License
MIT © 2025 [TamerOnLine](https://github.com/TamerOnLine)

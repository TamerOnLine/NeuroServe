import os
from dotenv import load_dotenv
load_dotenv()

print("HF_HOME          =", os.getenv("HF_HOME"))
print("TRANSFORMERS_CACHE =", os.getenv("TRANSFORMERS_CACHE"))
print("TORCH_HOME       =", os.getenv("TORCH_HOME"))

# محاولات قراءة فعلية من المكتبات (اختياري):
try:
    from huggingface_hub.constants import HF_HUB_CACHE
    print("HF_HUB_CACHE     =", HF_HUB_CACHE)
except Exception as e:
    print("HF hub not available:", e)

try:
    import torch
    print("torch.hub.get_dir() =", torch.hub.get_dir())
except Exception as e:
    print("torch hub not available:", e)

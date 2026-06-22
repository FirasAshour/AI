"""
config.py — shared configuration for the Real-ESRGAN super-resolution script.

Imported by sr_realesrgan.py so paths, the dataset slug, scale, and device
logic live in ONE place. Works unchanged on Windows CPU (smoke test) and
Colab GPU (full run).
"""

from pathlib import Path
import torch

# ----------------------------------------------------------------------
# 1. Dataset
# ----------------------------------------------------------------------
# Live Kaggle slug: 200 degraded portraits of men/women/kids.
KAGGLE_SLUG = "noobyogi0100/low-resolution-photographs"

# Process a small batch on CPU (smoke test); everything on GPU.
MAX_IMAGES = 8 if not torch.cuda.is_available() else None

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# ----------------------------------------------------------------------
# 2. Paths (absolute, anchored to this file — avoids Jupyter/CWD issues)
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent             # .../CV/SuperResolution
DATA_DIR = ROOT / "data"
LR_DIR = DATA_DIR / "lr"                            # local copy of inputs
SR_DIR = DATA_DIR / "sr"                            # super-res outputs
WEIGHTS_DIR = ROOT / "weights"                      # model weights cache

# Real-ESRGAN output folder.
SR_REALESRGAN_DIR = SR_DIR / "realesrgan"

for d in (LR_DIR, SR_REALESRGAN_DIR, WEIGHTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# 3. Super-resolution settings
# ----------------------------------------------------------------------
# Coupled to the downloaded weights: RealESRGAN_x4plus.pth is an x4 model.
# Do NOT change this in isolation — it must match the checkpoint you load.
SCALE = 4

# ----------------------------------------------------------------------
# 4. Device (one path, works on Windows CPU and Colab GPU)
# ----------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_FP16 = torch.cuda.is_available()   # half precision only helps on GPU

def list_images(folder: Path):
    """Recursively collect image paths under `folder`, sorted, capped by MAX_IMAGES."""
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    return files if MAX_IMAGES is None else files[:MAX_IMAGES]

def describe_env() -> str:
    dev = f"GPU ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU"
    cap = "all" if MAX_IMAGES is None else MAX_IMAGES
    return f"device={dev} | fp16={USE_FP16} | scale=x{SCALE} | images={cap}"
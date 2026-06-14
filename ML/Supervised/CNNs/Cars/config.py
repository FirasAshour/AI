import os
import torch

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = r"C:\Projects\ML\Supervised\CNNs\Cars\data\cars_dataset"
OUTPUT_DIR = "outputs"

# ── Data split ratios ──────────────────────────────────────────────────────────
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10

# ── Image settings ─────────────────────────────────────────────────────────────
IMG_SIZE = 300
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── Training hyperparameters ───────────────────────────────────────────────────
BATCH_SIZE  = 32      # use 32 on Colab GPU
NUM_EPOCHS  = 30      # use 30 on Colab GPU
LR          = 1e-3

# ── Hardware ───────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Model ──────────────────────────────────────────────────────────────────────
BACKBONE   = "efficientnet_b3"
PRETRAINED = True

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── ImageNet normalization stats ───────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ------------ config.py additions for Step 3 -----------------------------------
NUM_WORKERS = 0      # 0 for local Windows/Jupyter; bump to 2 on Colab
PIN_MEMORY  = True   # ignored on CPU, speeds transfer on GPU

# config.py additions for Step 5
HEAD_EPOCHS = 3            # stage 1: head-only, keep low for local smoke-testing
FINETUNE_EPOCHS = 10       # stage 2: full-network fine-tune
FINETUNE_LR_FACTOR = 0.1   # stage 2 LR = LR * this (e.g. 1e-3 -> 1e-4)

# ── Auto-scale for CPU ─────────────────────────────────────────────────────────
if DEVICE == "cpu":
    BATCH_SIZE = 8
    NUM_EPOCHS = 20



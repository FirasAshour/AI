import os
import torch

# ── Base directory (absolute, resolves regardless of working directory) ─────────
_HERE = os.path.dirname(os.path.abspath(__file__))

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = r"C:\Projects\ML\Supervised\CNNs\Cars\data\cars_dataset"
OUTPUT_DIR = os.path.join(_HERE, "outputs")

# ── Data split ratios ──────────────────────────────────────────────────────────
TRAIN_RATIO = 0.80
VAL_RATIO   = 0.10
TEST_RATIO  = 0.10

# ── Image settings ─────────────────────────────────────────────────────────────
IMG_SIZE = 300
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── Training hyperparameters ───────────────────────────────────────────────────
BATCH_SIZE  = 32
NUM_EPOCHS  = 30
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

# ── DataLoader settings ─────────────────────────────────────────────────────────
NUM_WORKERS = 0
PIN_MEMORY  = True

# ── Two-stage training schedule ─────────────────────────────────────────────────
HEAD_EPOCHS = 3
FINETUNE_EPOCHS = 10
FINETUNE_LR_FACTOR = 0.1

# ── Auto-scale for CPU ─────────────────────────────────────────────────────────
if DEVICE == "cpu":
    BATCH_SIZE = 8
    NUM_EPOCHS = 20
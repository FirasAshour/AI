"""
step1_data.py — Data preparation for the car classifier pipeline.

What this script does:
  1. Scans DATA_DIR for class folders and valid image files
  2. Builds a class-name list  →  outputs/class_names.txt
  3. Generates a summary CSV   →  outputs/dataset_summary.csv
  4. Plots class distribution  →  outputs/class_distribution.png
  5. Creates train/val/test splits (stratified per class)
  6. Saves a sample image grid →  outputs/sample_images.png

Run:
  python step1_data.py

Or import in later steps:
  from step1_data import run as prepare_data
  data = prepare_data()   # returns dict with splits + class_names
"""

import os
import random
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split

from config import (
    DATA_DIR, OUTPUT_DIR, IMG_EXTS,
    TRAIN_RATIO, VAL_RATIO,
    RANDOM_SEED,
)

warnings.filterwarnings("ignore")


# ── Helpers ────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def is_valid_image(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS


# ── 1. Scan dataset ────────────────────────────────────────────────────────────

def scan_dataset(data_dir: str) -> tuple[dict, list]:
    """
    Walk data_dir and collect image paths per class.

    Returns
    -------
    class_to_images : dict  { class_name: [abs_path, ...] }
    class_names     : sorted list of class names
    """
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {root.resolve()}\n"
            f"Update DATA_DIR in config.py or set the CAR_DATA_DIR env variable."
        )

    class_to_images = defaultdict(list)

    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        images = [str(p) for p in class_dir.rglob("*") if is_valid_image(p)]
        if images:
            class_to_images[class_dir.name] = sorted(images)

    if not class_to_images:
        raise ValueError(
            f"No class sub-folders with images found under: {root.resolve()}\n"
            "Expected layout:  DATA_DIR/<ClassName>/<image_files>"
        )

    class_names = sorted(class_to_images.keys())
    return dict(class_to_images), class_names


# ── 2. Save class names ────────────────────────────────────────────────────────

def save_class_names(class_names: list, output_dir: str) -> None:
    out = Path(output_dir) / "class_names.txt"
    out.write_text("\n".join(class_names))
    print(f"  ✓ class_names.txt     ({len(class_names)} classes)")


# ── 3. Summary CSV ─────────────────────────────────────────────────────────────

def build_summary_csv(
    class_to_images: dict,
    class_names: list,
    output_dir: str,
) -> pd.DataFrame:
    """
    Per-class summary: class_name, class_idx, num_images, pct_of_total
    """
    total = sum(len(v) for v in class_to_images.values())
    rows = [
        {
            "class_name":   name,
            "class_idx":    idx,
            "num_images":   len(class_to_images[name]),
            "pct_of_total": round(100 * len(class_to_images[name]) / total, 2),
        }
        for idx, name in enumerate(class_names)
    ]
    df = (
        pd.DataFrame(rows)
        .sort_values("num_images", ascending=False)
        .reset_index(drop=True)
    )
    df.to_csv(Path(output_dir) / "dataset_summary.csv", index=False)
    print(f"  ✓ dataset_summary.csv ({len(df)} rows)")
    return df


# ── 4. Class distribution plot ─────────────────────────────────────────────────

def plot_class_distribution(df: pd.DataFrame, output_dir: str, top_n: int = 40) -> None:
    """Bar chart of the top-N classes by image count."""
    plot_df   = df.head(top_n)
    n         = len(plot_df)
    fig, ax   = plt.subplots(figsize=(max(12, n * 0.35), 5))

    bars = ax.bar(range(n), plot_df["num_images"], color="#4A90D9",
                  edgecolor="white", linewidth=0.4)

    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(int(bar.get_height())),
            ha="center", va="bottom", fontsize=6, color="#444",
        )

    suffix = f" (top {top_n})" if len(df) > top_n else ""
    ax.set_xticks(range(n))
    ax.set_xticklabels(plot_df["class_name"], rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("Number of images")
    ax.set_title(f"Class distribution{suffix}", fontsize=12, pad=10)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(Path(output_dir) / "class_distribution.png", dpi=150)
    plt.close(fig)
    print("  ✓ class_distribution.png")


# ── 5. Train / val / test split ────────────────────────────────────────────────

def split_dataset(
    class_to_images: dict,
    class_names: list,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list, list, list]:
    """
    Stratified split into train / val / test.
    Each split is a list of (image_path, class_idx) tuples.

    The split happens per-class so every class appears in every split,
    even when the dataset is imbalanced.
    """
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    test_ratio   = 1.0 - train_ratio - val_ratio

    train_items, val_items, test_items = [], [], []

    for cls_name, paths in class_to_images.items():
        idx   = class_to_idx[cls_name]
        items = [(p, idx) for p in paths]

        if len(items) < 3:
            train_items.extend(items)
            continue

        train, remainder = train_test_split(
            items, test_size=(1.0 - train_ratio),
            random_state=seed, shuffle=True,
        )
        val_rel = val_ratio / (val_ratio + test_ratio)
        val, test = train_test_split(
            remainder, test_size=(1.0 - val_rel),
            random_state=seed, shuffle=True,
        )

        train_items.extend(train)
        val_items.extend(val)
        test_items.extend(test)

    rng = random.Random(seed)
    for split in (train_items, val_items, test_items):
        rng.shuffle(split)

    return train_items, val_items, test_items


# ── 6. Sample image grid ───────────────────────────────────────────────────────

def save_sample_grid(
    class_to_images: dict,
    class_names: list,
    output_dir: str,
    num_classes: int = 8,
    images_per_class: int = 4,
) -> None:
    """Grid of sample images: num_classes rows × images_per_class cols."""
    sampled = class_names[:num_classes]
    rows, cols = len(sampled), images_per_class

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    if rows == 1: axes = [axes]
    if cols == 1: axes = [[ax] for ax in axes]

    for r, cls_name in enumerate(sampled):
        pool   = class_to_images[cls_name]
        sample = random.sample(pool, min(cols, len(pool)))
        sample += [None] * (cols - len(sample))

        for c, img_path in enumerate(sample):
            ax = axes[r][c]
            ax.axis("off")
            if img_path and Path(img_path).exists():
                try:
                    ax.imshow(mpimg.imread(img_path))
                except Exception:
                    pass
            if c == 0:
                ax.set_title(cls_name, fontsize=7, loc="left", pad=2)

    plt.suptitle("Sample images per class", fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(Path(output_dir) / "sample_images.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ sample_images.png")


# ── Summary table ──────────────────────────────────────────────────────────────

def print_split_summary(train: list, val: list, test: list, num_classes: int) -> None:
    total = len(train) + len(val) + len(test)
    print(f"\n  {'Split':<8} {'Images':>8}  {'Share':>7}")
    print(f"  {'-' * 28}")
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        print(f"  {name:<8} {len(split):>8,}  {100 * len(split) / total:>6.1f}%")
    print(f"  {'Total':<8} {total:>8,}  {'100.0%':>7}")
    print(f"\n  Classes : {num_classes}")


# ── Main ───────────────────────────────────────────────────────────────────────

def run() -> dict:
    """
    Execute the full Step 1 pipeline.

    Returns
    -------
    {
        "class_names":     list[str],
        "class_to_images": dict[str, list[str]],
        "train":           list[tuple[str, int]],
        "val":             list[tuple[str, int]],
        "test":            list[tuple[str, int]],
    }
    Step 2 onwards imports and calls run() directly.
    """
    set_seed(RANDOM_SEED)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 50)
    print("  Step 1 — Data preparation")
    print("=" * 50)

    print(f"\n[1/5] Scanning: {Path(DATA_DIR).resolve()}")
    class_to_images, class_names = scan_dataset(DATA_DIR)
    total = sum(len(v) for v in class_to_images.values())
    print(f"      {len(class_names)} classes · {total:,} images")

    print("\n[2/5] Saving outputs …")
    save_class_names(class_names, OUTPUT_DIR)

    print("\n[3/5] Building summary CSV …")
    df = build_summary_csv(class_to_images, class_names, OUTPUT_DIR)

    print("\n[4/5] Plotting class distribution …")
    plot_class_distribution(df, OUTPUT_DIR)

    print("\n[5/5] Splitting + sample grid …")
    train, val, test = split_dataset(
        class_to_images, class_names,
        TRAIN_RATIO, VAL_RATIO, RANDOM_SEED,
    )
    save_sample_grid(class_to_images, class_names, OUTPUT_DIR)
    print_split_summary(train, val, test, len(class_names))

    print("\n" + "=" * 50)
    print(f"  Done. Outputs → {Path(OUTPUT_DIR).resolve()}")
    print("=" * 50 + "\n")

    return {
        "class_names":     class_names,
        "class_to_images": class_to_images,
        "train":           train,
        "val":             val,
        "test":            test,
    }


if __name__ == "__main__":
    run()
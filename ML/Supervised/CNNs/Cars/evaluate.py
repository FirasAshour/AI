# evaluate.py
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import config
from dataset import build_dataloaders
from model import build_model


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        preds = outputs.argmax(1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())
    return np.concatenate(all_labels), np.concatenate(all_preds)


def evaluate_model(train_df, val_df, test_df, class_names, checkpoint="best_model.pth"):
    device = torch.device(config.DEVICE)
    num_classes = len(class_names)

    # rebuild loaders; we only use the test loader here
    _, _, test_loader = build_dataloaders(train_df, val_df, test_df)

    model = build_model(num_classes, pretrained=False).to(device)
    ckpt_path = os.path.join(config.OUTPUT_DIR, checkpoint)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    y_true, y_pred = collect_predictions(model, test_loader, device)

    # ---- per-class metrics ----
    report_dict = classification_report(
        y_true, y_pred,
        labels=list(range(num_classes)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_path = os.path.join(config.OUTPUT_DIR, "classification_report.csv")
    report_df.to_csv(report_path)
    print(f"  ✓ classification_report.csv")

    # text summary to console
    print("\n" + classification_report(
        y_true, y_pred,
        labels=list(range(num_classes)),
        target_names=class_names,
        zero_division=0,
    ))

    # ---- confusion matrix ----
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    save_confusion_matrix(cm, class_names, config.OUTPUT_DIR)

    overall_acc = (y_true == y_pred).mean()
    print(f"\n  Overall test accuracy: {overall_acc:.4f}")
    return report_df, cm


def save_confusion_matrix(cm, class_names, output_dir):
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.4), max(6, n * 0.4)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=90)
    ax.set_title("Confusion matrix (test set)")
    plt.tight_layout()
    out = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  ✓ confusion_matrix.png")
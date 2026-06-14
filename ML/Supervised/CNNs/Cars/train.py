# train.py
import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import config
from dataset import build_dataloaders
from model import build_model, freeze_backbone, unfreeze_all, count_trainable_params


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def fit(train_df, val_df, test_df, num_classes):
    device = torch.device(config.DEVICE)
    train_loader, val_loader, test_loader = build_dataloaders(train_df, val_df, test_df)
    model = build_model(num_classes, pretrained=config.PRETRAINED).to(device)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_path = os.path.join(config.OUTPUT_DIR, "best_model.pth")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    history = []

    # ---- Stage 1: train head only ----
    freeze_backbone(model)
    trainable, total = count_trainable_params(model)
    print(f"[Stage 1] Head only: {trainable:,} / {total:,} trainable")
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.HEAD_EPOCHS)

    for epoch in range(config.HEAD_EPOCHS):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        history.append(("stage1", epoch, tr_loss, tr_acc, va_loss, va_acc))
        print(f"[S1 {epoch+1}/{config.HEAD_EPOCHS}] "
              f"train {tr_loss:.4f}/{tr_acc:.4f} | val {va_loss:.4f}/{va_acc:.4f} | {time.time()-t0:.0f}s")
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), best_path)

    # ---- Stage 2: fine-tune whole network ----
    unfreeze_all(model)
    trainable, total = count_trainable_params(model)
    print(f"[Stage 2] Full network: {trainable:,} / {total:,} trainable")
    optimizer = AdamW(model.parameters(), lr=config.LR * config.FINETUNE_LR_FACTOR)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.FINETUNE_EPOCHS)

    for epoch in range(config.FINETUNE_EPOCHS):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        history.append(("stage2", epoch, tr_loss, tr_acc, va_loss, va_acc))
        print(f"[S2 {epoch+1}/{config.FINETUNE_EPOCHS}] "
              f"train {tr_loss:.4f}/{tr_acc:.4f} | val {va_loss:.4f}/{va_acc:.4f} | {time.time()-t0:.0f}s")
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), best_path)

    print(f"\nBest val acc: {best_val_acc:.4f} (saved to {best_path})")
    model.load_state_dict(torch.load(best_path, map_location=device))
    te_loss, te_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test: {te_loss:.4f} loss / {te_acc:.4f} acc")
    return model, history
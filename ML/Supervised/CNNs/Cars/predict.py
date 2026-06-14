# predict.py
import os
import cv2
import torch
import torch.nn.functional as F

import config
from model import build_model
from transforms import get_eval_transforms


def load_class_names(path=None):
    if path is None:
        path = os.path.join(config.OUTPUT_DIR, "class_names.txt")
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_trained_model(class_names, checkpoint="best_model.pth", device=None):
    if device is None:
        device = torch.device(config.DEVICE)
    model = build_model(len(class_names), pretrained=False).to(device)
    ckpt_path = os.path.join(config.OUTPUT_DIR, checkpoint)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model, device


@torch.no_grad()
def predict_image(image_path, model, class_names, device, top_k=3):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = get_eval_transforms()
    tensor = transform(image=image)["image"]      # CHW float tensor
    tensor = tensor.unsqueeze(0).to(device)        # add batch dim -> 1CHW

    logits = model(tensor)
    probs = F.softmax(logits, dim=1)[0]            # drop batch dim

    k = min(top_k, len(class_names))
    top_probs, top_idxs = probs.topk(k)

    results = [
        (class_names[idx], float(prob))
        for idx, prob in zip(top_idxs.cpu().numpy(), top_probs.cpu().numpy())
    ]
    return results


def predict(image_path, top_k=3):
    """Convenience one-shot: loads everything, predicts a single image."""
    class_names = load_class_names()
    model, device = load_trained_model(class_names)
    return predict_image(image_path, model, class_names, device, top_k=top_k)
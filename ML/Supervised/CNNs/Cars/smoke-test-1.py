import os
import torch
import config
from model import build_model, freeze_backbone, count_trainable_params

class_names_path = os.path.join(config.OUTPUT_DIR, "class_names.txt")
with open(class_names_path) as f:
    num_classes = len([line for line in f.read().splitlines() if line.strip()])

model = build_model(num_classes, pretrained=config.PRETRAINED)

dummy = torch.randn(2, 3, config.IMG_SIZE, config.IMG_SIZE)
out = model(dummy)
print(out.shape)   # expect [2, num_classes]

freeze_backbone(model)
trainable, total = count_trainable_params(model)
print(f"Frozen: {trainable:,} / {total:,} trainable")
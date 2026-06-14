# model.py
import torch
import torch.nn as nn
import timm

import config


def build_model(num_classes, pretrained=True):
    model = timm.create_model(
        config.BACKBONE,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model


def freeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.get_classifier().parameters():
        param.requires_grad = True
    return model


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True
    return model


def count_trainable_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
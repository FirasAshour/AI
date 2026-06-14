# transforms.py
import albumentations as A
from albumentations.pytorch import ToTensorV2
import config

MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

IMG_SIZE = getattr(config, "IMG_SIZE", 300)


def get_train_transforms():
    return A.Compose([
        A.RandomResizedCrop(
            size=(IMG_SIZE, IMG_SIZE),
            scale=(0.8, 1.0),
            ratio=(0.9, 1.111),
        ),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.1, rotate_limit=15,
            border_mode=0, p=0.5,
        ),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10),
        ], p=0.5),
        A.CoarseDropout(
            num_holes_range=(1, 8),
            hole_height_range=(0.05, 0.1),
            hole_width_range=(0.05, 0.1),
            fill=0,
            p=0.3,
        ),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def get_eval_transforms():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])
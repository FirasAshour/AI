# dataset.py
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

import config
from transforms import get_train_transforms, get_eval_transforms


class CarDataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["filepath"]
        label = int(row["label_idx"])

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        return image, label


def build_dataloaders(train_df, val_df, test_df):
    train_ds = CarDataset(train_df, transforms=get_train_transforms())
    val_ds   = CarDataset(val_df,   transforms=get_eval_transforms())
    test_ds  = CarDataset(test_df,  transforms=get_eval_transforms())

    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, drop_last=False,
    )
    return train_loader, val_loader, test_loader
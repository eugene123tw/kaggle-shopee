from typing import Union, List, Any, Optional

import albumentations
import numpy as np
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from utils import (
    read_csv,
    ShopeeDataset,
    collate
)


class ShopeeTrainValDataModule(LightningDataModule):
    def __init__(self, hparams, input_type: str):
        super(ShopeeTrainValDataModule, self).__init__()
        self.hparams = hparams
        self.input_type = input_type

    def setup(self, stage: Optional[str] = None):
        if stage == 'test':
            return
        lines = read_csv(self.hparams.label_csv)
        lines = lines[np.argsort(lines[:, -1])]
        label_map = {label: i for i, label in enumerate(np.unique(np.array(lines)[:, 4]))}
        kf = KFold(n_splits=5)

        for i, (train_indices, val_indices) in enumerate(kf.split(range(len(lines)))):
            if i == self.hparams.fold:
                train_lines, val_lines = lines[train_indices], lines[val_indices]
                break

        self.train_dataset = ShopeeDataset(
            self.hparams,
            lines=train_lines,
            label_map=label_map,
            transform=self.train_transforms(),
            input_type=self.input_type
        )

        self.val_dataset = ShopeeDataset(
            self.hparams,
            lines=val_lines,
            label_map=label_map,
            transform=self.val_transforms(),
            input_type=self.input_type
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=collate
        )

    def train_dataloader(self) -> Any:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            collate_fn=collate
        )

    def train_transforms(self):
        if self.input_type == 'text':
            return None
        transform = albumentations.Compose([
            albumentations.Resize(self.hparams.input_size, self.hparams.input_size),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.RandomBrightnessContrast(p=0.5, brightness_limit=0.3, contrast_limit=0.3),
            albumentations.HueSaturationValue(p=0.5),
            albumentations.ShiftScaleRotate(
                p=0.5,
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=20),
            albumentations.CoarseDropout(p=0.5),
            albumentations.Normalize(),
            ToTensorV2()
        ])
        return transform

    def val_transforms(self):
        if self.input_type == 'text':
            return None
        transform = albumentations.Compose([
            albumentations.Resize(self.hparams.input_size, self.hparams.input_size),
            albumentations.Normalize(),
            ToTensorV2()
        ])
        return transform

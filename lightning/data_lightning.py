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
    ShopeeTestDataset,
    collate
)


class ShopeeTrainValDataModule(LightningDataModule):
    def __init__(self, hparams):
        super(ShopeeTrainValDataModule, self).__init__()
        self.hparams = hparams

    def setup(self, stage: Optional[str] = None):
        lines = read_csv(self.hparams.label_csv)
        label_map = {label: i for i, label in enumerate(np.unique(np.array(lines)[:, 4]))}

        kf = KFold(n_splits=5, shuffle=True, random_state=self.hparams.random_seed)
        for i, (train_indices, val_indices) in enumerate(kf.split(range(len(lines)))):
            if i == self.hparams.fold:
                train_lines, val_lines = lines[train_indices], lines[val_indices]
                break

        self.train_dataset = ShopeeDataset(
            self.hparams,
            lines=train_lines,
            label_map=label_map,
            transform=self.train_transforms(),
        )

        self.val_dataset = ShopeeDataset(
            self.hparams,
            lines=val_lines,
            label_map=label_map,
            transform=self.val_transforms(),
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
        transform = albumentations.Compose([
            albumentations.Resize(self.hparams.input_size, self.hparams.input_size),
            albumentations.CenterCrop(height=self.hparams.center_crop, width=self.hparams.center_crop),
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
            ToTensorV2(),
        ])
        return transform

    def val_transforms(self):
        transform = albumentations.Compose([
            albumentations.Resize(self.hparams.input_size, self.hparams.input_size),
            albumentations.CenterCrop(height=self.hparams.center_crop, width=self.hparams.center_crop),
            albumentations.Normalize(),
            ToTensorV2(),
        ])
        return transform


class ShopeeTestDataModule(LightningDataModule):
    def __init__(self, hparams):
        super(ShopeeTestDataModule, self).__init__()
        self.hparams = hparams

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        test_lines = read_csv(self.hparams.test_csv)
        test_dataset = ShopeeTestDataset(
            self.hparams,
            lines=test_lines,
            transform=self.test_transforms()
        )
        return DataLoader(
            test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers
        )

    def test_transforms(self):
        transform = albumentations.Compose([
            albumentations.Resize(self.hparams.input_size, self.hparams.input_size),
            albumentations.CenterCrop(height=self.hparams.center_crop, width=self.hparams.center_crop),
            albumentations.Normalize(),
            ToTensorV2(),
        ])

        return transform

import csv
from typing import Optional, Union, List

from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from utils.dataset import ShopeeDataset


class ShopeeDataModule(LightningDataModule):
    def __init__(self, hparams: DictConfig):
        super(ShopeeDataModule, self).__init__()
        self.hparams = hparams

    def setup(self, stage: Optional[str] = None):
        lines = []
        label_set = set()
        with open(self.hparams.label_csv, "r") as f:
            reader = csv.reader(f)
            for i, line in enumerate(reader):
                if i > 0:
                    lines.append(line)
                    label_set.add(line[4])
        label_map = {label: i for i, label in enumerate(list(label_set))}

        train_lines, test_lines = train_test_split(lines, test_size=0.2, random_state=42)

        self.train_dataset = ShopeeDataset(
            self.hparams, label_map, train_lines,
            transform=transforms.Compose([
                transforms.ToTensor(),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(10),
                transforms.Resize((self.hparams.input_size, self.hparams.input_size)),
            ])
        )

        self.val_dataset = ShopeeDataset(
            self.hparams, label_map, test_lines,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.hparams.input_size, self.hparams.input_size)),
            ])
        )

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size)

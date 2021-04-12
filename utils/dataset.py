import os
from typing import Callable, Optional

import cupy as cp
import cv2
from torch.utils.data import Dataset

from .utils import build_gt


class ShopeeDataset(Dataset):
    def __init__(self, hparams, label_map, lines: cp.ndarray, transform: Optional[Callable] = None):
        super(ShopeeDataset, self).__init__()
        self.hparams = hparams
        self.label_map = label_map
        self.lines = lines
        self.transform = transform
        self.gt = build_gt(lines)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, item):
        line = self.lines[item]
        fname = line[0]
        sentence = line[3]
        label = self.label_map[line[-1]]
        gt = self.gt[fname]

        img_path = os.path.join(self.hparams.train_dir, line[1])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            res = self.transform(image=img)
            img = res['image']

        return fname, img, sentence, label, gt


class ShopeeTestDataset(Dataset):
    def __init__(self, hparams, lines: cp.ndarray, transform: Optional[Callable] = None):
        super(ShopeeTestDataset, self).__init__()
        self.hparams = hparams
        self.lines = lines
        self.transform = transform

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, item):
        line = self.lines[item]
        fname = line[0]
        phash = line[2]
        sentence = line[3]

        img_path = os.path.join(self.hparams.test_dir, line[1])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            res = self.transform(image=img)
            img = res['image']
        return fname, img, sentence, phash

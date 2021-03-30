import os
from typing import Callable, Optional

import cv2
import numpy as np
from torch.utils.data import Dataset

from utils.utils import build_gt


class ShopeeDataset(Dataset):
    def __init__(self, hparams, label_map, lines: np.ndarray, transform: Optional[Callable] = None):
        super(ShopeeDataset, self).__init__()
        self.hparams = hparams
        self.label_map = label_map
        self.lines = lines
        self.transform = transform
        self.gt = build_gt(self.lines)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, item):
        line = self.lines[item]
        fname = line[0]
        sentence = line[3]
        label = self.label_map[line[-1]]

        img_path = os.path.join(self.hparams.train_dir, line[1])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            res = self.transform(image=img)
            img = res['image']

        return fname, img, sentence, label


class ShopeeTestDataset(Dataset):
    def __init__(self, hparams, lines: np.ndarray, transform: Optional[Callable] = None):
        super(ShopeeTestDataset, self).__init__()
        self.hparams = hparams
        self.lines = lines
        self.transform = transform

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, item):
        line = self.lines[item]
        fname = line[0]
        sentence = line[3]

        img_path = os.path.join(self.hparams.test_dir, line[1])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            res = self.transform(image=img)
            img = res['image']
        return fname, img, sentence

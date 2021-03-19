import os
from typing import List, Callable, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from utils.utils import build_gt


class ShopeeDataset(Dataset):
    def __init__(self, hparams, lines: List, transform: Optional[Callable] = None):
        super(ShopeeDataset, self).__init__()
        self.hparams = hparams
        self.label_map = {label: i for i, label in enumerate(np.unique(np.array(lines)[:, 4]))}
        self.lines = lines
        self.transform = transform
        self.gt = build_gt(self.lines)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, item):
        line = self.lines[item]
        fname = line[0]
        label = self.label_map[line[-1]]

        img_path = os.path.join(self.hparams.train_dir, line[1])
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)
        return fname, img, label

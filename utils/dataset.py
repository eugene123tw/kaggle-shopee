import os
from typing import Dict, List, Callable, Optional

import cv2
from torch.utils.data import Dataset


class ShopeeDataset(Dataset):
    def __init__(self, hparams, label_map: Dict, lines: List, transform: Optional[Callable] = None):
        super(ShopeeDataset, self).__init__()
        self.hparams = hparams
        self.label_map = label_map
        self.lines = lines
        self.transform = transform

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, item):
        line = self.lines[item]

        label = self.label_map[line[-1]]

        img_path = os.path.join(self.hparams.train_dir, line[1])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            img = self.transform(img)
        return img, label

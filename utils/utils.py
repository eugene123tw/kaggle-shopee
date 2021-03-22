import csv
from collections import OrderedDict, Counter
from typing import List, Dict

import numpy as np
import torch


def read_csv(path) -> List:
    lines = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        for i, line in enumerate(reader):
            if i > 0:
                lines.append(line)
    return lines


def build_gt(lines: List) -> Dict:
    gt = OrderedDict()
    for i, line in enumerate(lines):
        posting_id = line[0]
        label_group = line[-1]
        gt[posting_id] = list(np.where(lines[:, -1] == label_group)[0])
    return gt


def get_class_weights(lines: np.ndarray, label_map: Dict[str, int], n_classes: int) -> torch.Tensor:
    counter = Counter(lines[:, -1])
    counts = np.zeros((n_classes))
    for label, count in counter.items():
        counts[label_map[label]] = count
    class_weights = 1 / np.log1p(counts)
    class_weights = (class_weights / class_weights.sum()) * n_classes
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    return class_weights

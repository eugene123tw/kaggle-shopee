import csv
from typing import List

import numpy as np


def read_csv(path) -> List:
    lines = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        for i, line in enumerate(reader):
            if i > 0:
                lines.append(line)
    return lines


def build_gt(lines: List) -> np.ndarray:
    gt = np.zeros((len(lines), len(lines)))
    groups = np.array(lines)[:, -1]

    for i, line in enumerate(lines):
        gt[i, np.where(line[-1] == groups)[0]] = 1
    gt = np.triu(gt)
    return gt

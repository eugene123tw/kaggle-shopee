import csv
from typing import List


def read_csv(path) -> List:
    lines = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        for i, line in enumerate(reader):
            if i > 0:
                lines.append(line)
    return lines

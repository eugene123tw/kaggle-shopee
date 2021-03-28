import csv
import os
from collections import OrderedDict, Counter
from typing import List, Dict

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity


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


def compute_cosine_similarity(embedding_dic: Dict[str, np.ndarray]) -> np.ndarray:
    fnames, embeddings = list(embedding_dic.keys()), list(embedding_dic.values())
    sim_matrix = cosine_similarity(embeddings)
    sim_matrix = (sim_matrix - np.min(sim_matrix)) / (np.max(sim_matrix) - np.min(sim_matrix))
    return sim_matrix


def write_submission(submit_dict, path='.'):
    with open(os.path.join(path, 'submission.csv'), "w") as f:
        writer = csv.writer(f)
        writer.writerow(['posting_id', 'matches'])
        writer.writerows(zip(submit_dict['posting_id'], submit_dict['matches']))

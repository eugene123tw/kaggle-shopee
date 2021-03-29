import csv
import os
from collections import OrderedDict, Counter
from typing import List, Dict

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


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


def cosine_similarity_chunk(embeddings: np.ndarray, threshold: float) -> np.ndarray:
    pred_indices = []
    chunk = 1024 * 2

    counter = len(embeddings) // chunk
    if len(embeddings) % chunk != 0: counter += 1
    for j in range(counter):
        a = j * chunk
        b = (j + 1) * chunk
        b = min(b, len(embeddings))

        sim_matrix = np.matmul(embeddings, embeddings[a:b].T).T
        for k in range(b - a):
            indices = np.where(sim_matrix[k,] > threshold)[0]
            pred_indices.append(indices)
    return np.array(pred_indices)


def compute_cosine_similarity(embedding_dic: Dict[str, np.ndarray], batch_compute: bool = False,
                              threshold: float = 0.5) -> np.ndarray:
    embeddings = list(embedding_dic.values())
    embeddings = normalize(embeddings)
    if not batch_compute:
        sim_matrix = cosine_similarity(embeddings)
        pred_indices = []
        for sim in sim_matrix:
            indices = np.where(sim > threshold)[0]
            pred_indices.append(indices)
        return np.array(pred_indices)
    return cosine_similarity_chunk(embeddings, threshold)


def write_submission(submit_dict, path='.'):
    with open(os.path.join(path, 'submission.csv'), "w") as f:
        writer = csv.writer(f)
        writer.writerow(['posting_id', 'matches'])
        writer.writerows(zip(submit_dict['posting_id'], submit_dict['matches']))

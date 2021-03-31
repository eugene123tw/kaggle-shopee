import csv
import os
from collections import OrderedDict, Counter
from typing import Dict

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


def read_csv(path) -> np.ndarray:
    lines = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        for i, line in enumerate(reader):
            if i > 0:
                lines.append(line)
    return np.array(lines)


def build_gt(lines: np.ndarray) -> Dict:
    gt = OrderedDict()
    fnames = lines[:, 0]
    for i, line in enumerate(lines):
        posting_id = line[0]
        label_group = line[-1]
        gt[posting_id] = fnames[np.where(lines[:, -1] == label_group)[0]]
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


def cosine_similarity_chunk(fnames, embeddings: np.ndarray, threshold: float) -> np.ndarray:
    pred_fnames = []
    chunk = 1024 * 2
    embeddings = normalize(embeddings)
    counter = len(embeddings) // chunk
    if len(embeddings) % chunk != 0:
        counter += 1
    for j in range(counter):
        a = j * chunk
        b = (j + 1) * chunk
        b = min(b, len(embeddings))

        sim_matrix = np.matmul(embeddings, embeddings[a:b].T).T
        for k in range(b - a):
            pred_fnames.append(fnames[np.where(sim_matrix[k,] > threshold)[0]])
    return np.array(pred_fnames)


def compute_cosine_similarity(embeddings, fnames, batch_compute: bool = False,
                              threshold: float = 0.9, top_k=10) -> np.ndarray:
    if not batch_compute:
        sim_matrix = cosine_similarity(embeddings)
        pred_fnames = []
        for sim in sim_matrix:
            pred_fnames.append(fnames[np.where(sim > threshold)[0]])
        return np.array(pred_fnames)
    return cosine_similarity_chunk(fnames, embeddings, threshold)


def write_submission(submit_dict, path='.'):
    with open(os.path.join(path, 'submission.csv'), "w") as f:
        writer = csv.writer(f)
        writer.writerow(['posting_id', 'matches'])
        writer.writerows(zip(submit_dict['posting_id'], submit_dict['matches']))

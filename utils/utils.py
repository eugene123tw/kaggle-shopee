import csv
import os
from collections import OrderedDict, Counter
from typing import Dict, List

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


def cosine_similarity_chunk(fnames, embeddings: np.ndarray, threshold: float, top_k: int) -> Dict:
    pred_fnames = {}
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
            match_indices = np.where(sim_matrix[k,] > threshold)[0]
            if len(match_indices) > top_k:
                match_indices = np.argsort(sim_matrix[k,])[-top_k:]
            pred_fnames[fnames[a + k]] = fnames[match_indices]
    return pred_fnames


def compute_cosine_similarity(embeddings, fnames, batch_compute: bool = False, threshold: float = 0.5,
                              top_k: int = 50) -> Dict:
    if not batch_compute:
        sim_matrix = cosine_similarity(embeddings)
        pred_fnames = {}
        for i, sim in enumerate(sim_matrix):
            match_indices = np.where(sim > threshold)[0]
            if len(match_indices) > top_k:
                match_indices = np.argsort(sim)[-top_k:]
            pred_fnames[fnames[i]] = fnames[match_indices]
        return pred_fnames
    return cosine_similarity_chunk(fnames, embeddings, threshold, top_k)


def compute_f1_score(pred_dict: Dict, gt: Dict) -> float:
    TP, FP, FN = 0, 0, 0
    for fname, gt_fnames in gt.items():
        pred_fnames = pred_dict[fname]
        TP += len(set(gt_fnames).intersection(set(pred_fnames)))
        FP += len(set(pred_fnames) - set(gt_fnames))
        FN += len(set(gt_fnames) - set(pred_fnames))

    f1_value = TP / (TP + 0.5 * (FP + FN))
    return f1_value


def combine_pred_dicts(predictions: List[Dict]) -> Dict:
    combined_dict = {}
    for pred_dict in predictions:
        for fname, pred in pred_dict.items():
            if fname in combined_dict:
                combined_dict[fname] = np.intersect1d(combined_dict[fname], pred)
            else:
                combined_dict[fname] = pred
    return combined_dict


def write_submission(submit_dict, path='.'):
    with open(os.path.join(path, 'submission.csv'), "w") as f:
        writer = csv.writer(f)
        writer.writerow(['posting_id', 'matches'])
        for fname, pred in submit_dict.items():
            writer.writerow([fname, pred])


def collate(batch):
    fnames = [item[0] for item in batch]
    imgs = torch.stack([item[1] for item in batch], 0)
    sentences = [item[2] for item in batch]
    labels = torch.tensor([item[3] for item in batch])
    gt = [item[4] for item in batch]
    return fnames, imgs, sentences, labels, gt

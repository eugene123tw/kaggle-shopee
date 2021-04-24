import csv
import os
import random
from collections import OrderedDict, Counter
from typing import Dict, List, Tuple, Union

import cupy as cp
import numpy as np
import torch
from cuml.neighbors import NearestNeighbors


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


def compute_cosine_similarity(fnames: np.array, embeddings: cp.ndarray, threshold: float, top_k: int,
                              get_prob=False) -> Tuple[Dict, Union[np.ndarray, List]]:
    if not isinstance(embeddings, cp.ndarray):
        raise NotImplemented("WRONG INPUT FORMAT")

    prob_array = np.zeros((len(fnames), len(fnames))) if get_prob else []

    pred_fnames = OrderedDict()
    chunk = 1024 * 2
    counter = len(embeddings) // chunk
    if len(embeddings) % chunk != 0:
        counter += 1
    for j in range(counter):
        a = j * chunk
        b = (j + 1) * chunk
        b = min(b, len(embeddings))

        sim_matrix = cp.matmul(embeddings, embeddings[a:b].T).T

        if get_prob:
            prob_array[a:b] = sim_matrix.get()

        for k in range(b - a):
            match_indices = cp.where(sim_matrix[k,] > threshold)[0]
            if len(match_indices) > top_k:
                match_indices = cp.argsort(sim_matrix[k,])[-top_k:]
            pred_fnames[fnames[a + k]] = fnames[match_indices.get()]
    return pred_fnames, prob_array


def compute_cosine_similarity_np(fnames: np.array, embeddings: np.ndarray, threshold: float, top_k: int) -> Dict:
    if not isinstance(fnames, np.ndarray) or not isinstance(embeddings, np.ndarray):
        raise NotImplemented("WRONG INPUT FORMAT")

    pred_fnames = {}
    chunk = 1024 * 2
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


def knn_similarity(embeddings: cp.ndarray, fnames: np.array, n_neighbors: int, threshold: float) -> Dict:
    knn = NearestNeighbors(n_neighbors=n_neighbors)
    knn.fit(embeddings)
    distances, indices = knn.kneighbors(embeddings)

    result = {}
    for i in range(embeddings.shape[0]):
        idx = np.where(distances[i,] < threshold)[0]
        ids = indices[i, idx]
        result[fnames[i]] = fnames[ids.get()]
    return result


def compute_f1_score(pred_dict: Dict, gt: Dict) -> float:
    TP, FP, FN = 0, 0, 0
    for fname, gt_fnames in gt.items():
        pred_fnames = pred_dict[fname]
        TP += len(set(gt_fnames).intersection(set(pred_fnames)))
        FP += len(set(pred_fnames) - set(gt_fnames))
        FN += len(set(gt_fnames) - set(pred_fnames))

    f1_value = TP / (TP + 0.5 * (FP + FN))
    return f1_value


def combine_pred_dicts(predictions: List[Dict], method='inter') -> Dict:
    """ Combine prediction dictionaries

    :param predictions:
    :param method: 'inter'/'union'
    :return:
    """
    if len(predictions) == 1:
        return predictions[0]

    combined_dict = {}
    for pred_dict in predictions:
        for fname, pred in pred_dict.items():
            if fname in combined_dict:
                if method == 'inter':
                    combined_dict[fname] = np.intersect1d(combined_dict[fname], pred)
                elif method == 'union':
                    combined_dict[fname] = np.unique(np.concatenate((combined_dict[fname], pred)))
                else:
                    raise NotImplemented("UNKNOWN COMBINE METHOD")

            else:
                combined_dict[fname] = pred
    return combined_dict


def write_submission(submit_dict, path='.'):
    with open(os.path.join(path, 'submission.csv'), "w") as f:
        writer = csv.writer(f)
        writer.writerow(['posting_id', 'matches'])
        for fname, pred in submit_dict.items():
            pred = ' '.join(pred)
            writer.writerow([fname, pred])


def collate(batch):
    fnames = [item[0] for item in batch]
    imgs = torch.stack([item[1] for item in batch], 0)
    sentences = [item[2] for item in batch]
    labels = torch.tensor([item[3] for item in batch])
    gt = [item[4] for item in batch]
    return fnames, imgs, sentences, labels, gt


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

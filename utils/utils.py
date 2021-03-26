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


def mask_fill(
    fill_value: float,
    tokens: torch.tensor,
    embeddings: torch.tensor,
    padding_index: int,
) -> torch.tensor:
    """
    Function that masks embeddings representing padded elements.
    :param fill_value: the value to fill the embeddings belonging to padded tokens.
    :param tokens: The input sequences [bsz x seq_len].
    :param embeddings: word embeddings [bsz x seq_len x hiddens].
    :param padding_index: Index of the padding token.
    """
    padding_mask = tokens.eq(padding_index).unsqueeze(-1)
    return embeddings.float().masked_fill_(padding_mask, fill_value).type_as(embeddings)


def lengths_to_mask(*lengths, **kwargs):
    """ Given a list of lengths, create a batch mask.

    Example:
        >>> lengths_to_mask([1, 2, 3])
        tensor([[ True, False, False],
                [ True,  True, False],
                [ True,  True,  True]])
        >>> lengths_to_mask([1, 2, 2], [1, 2, 2])
        tensor([[[ True, False],
                 [False, False]],
        <BLANKLINE>
                [[ True,  True],
                 [ True,  True]],
        <BLANKLINE>
                [[ True,  True],
                 [ True,  True]]])

    Args:
        *lengths (list of int or torch.Tensor)
        **kwargs: Keyword arguments passed to ``torch.zeros`` upon initially creating the returned
          tensor.

    Returns:
        torch.BoolTensor
    """
    # Squeeze to deal with random additional dimensions
    lengths = [l.squeeze().tolist() if torch.is_tensor(l) else l for l in lengths]

    # For cases where length is a scalar, this needs to convert it to a list.
    lengths = [l if isinstance(l, list) else [l] for l in lengths]
    assert all(len(l) == len(lengths[0]) for l in lengths)
    batch_size = len(lengths[0])
    other_dimensions = tuple([int(max(l)) for l in lengths])
    mask = torch.zeros(batch_size, *other_dimensions, **kwargs)
    for i, length in enumerate(zip(*tuple(lengths))):
        mask[i][[slice(int(l)) for l in length]].fill_(1)
    return mask.bool()
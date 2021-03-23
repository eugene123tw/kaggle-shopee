import os
from collections import Counter

import cv2
import matplotlib.pylab as plt
import numpy as np

from utils import read_csv

DATA_DIR = "/home/yuchunli/_DATASET/shopee-product-matching/train_images"


def plot_images(lines, column_idx, value):
    '''
    Plot images using image_path, based on the column & value filter
    '''
    plt.figure(figsize=(30, 30))
    value_filter = lines[lines[:, column_idx] == value]
    posting_id = value_filter[:, 0]
    image_paths = value_filter[:, 1]

    print(f'Total images: {len(image_paths)}')
    for i, j in enumerate(zip(image_paths, posting_id)):
        plt.subplot(10, 10, i + 1)
        img = cv2.cvtColor(cv2.imread(os.path.join(DATA_DIR, j[0])), cv2.COLOR_BGR2RGB)
        plt.title(j[1])
        plt.axis("off")
        plt.tight_layout()
        plt.imshow(img)


def query(lines: np.ndarray, column_idx, value):
    return lines[lines[:, column_idx] == value]


def same_hash_different_label(lines, phash_counter):
    for phash, count in phash_counter.items():
        if count > 2 and len(set(query(lines, 2, phash)[:, -1])) > 1:
            print(set(query(lines, 2, phash)[:, -1]))
            plot_images(lines, 2, phash)
            plt.show()

if __name__ == "__main__":
    # ['posting_id', 'image name', 'phash', 'title', 'label_group']
    lines = read_csv("/home/yuchunli/_DATASET/shopee-product-matching/train.csv")
    lines = np.array(lines)
    label_counter = Counter(lines[:, -1])
    phash_counter = Counter(lines[:, 2])
    same_hash_different_label(lines, phash_counter)


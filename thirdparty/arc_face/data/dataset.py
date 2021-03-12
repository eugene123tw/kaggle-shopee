import os

import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms as T


class Dataset(data.Dataset):

    def __init__(self, root, data_list_file, phase='train', input_shape=(1, 128, 128)):
        self.root = root
        self.phase = phase
        self.input_shape = input_shape

        with open(os.path.join(data_list_file), 'r') as fd:
            imgs = fd.readlines()

        imgs = [img.strip() for img in imgs]
        unique_dirs = np.unique([os.path.dirname(img) for img in imgs])
        self.label_mapping = {fname: i for i, fname in enumerate(unique_dirs)}
        self.imgs = np.random.permutation(imgs)

        # normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
        #                         std=[0.5, 0.5, 0.5])

        normalize = T.Normalize(mean=[0.5], std=[0.5])

        if self.phase == 'train':
            self.transforms = T.Compose([
                T.RandomCrop(self.input_shape[1:]),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = T.Compose([
                T.CenterCrop(self.input_shape[1:]),
                T.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        img_path = self.imgs[index]
        data = Image.open(os.path.join(self.root, img_path))
        data = data.convert('L')
        data = self.transforms(data)
        label = self.label_mapping[os.path.dirname(img_path)]
        return data.float(), label

    def __len__(self):
        return len(self.imgs)

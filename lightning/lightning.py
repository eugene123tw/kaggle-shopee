from typing import Union, List, Any

import numpy as np
import torch
from pytorch_lightning import LightningModule
from sklearn.metrics.pairwise import cosine_similarity
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from models.meta_model import Backbone
from utils import read_csv, dice, ShopeeDataset
from utils.loss import SphereProduct


class ShopeeLightning(LightningModule):
    def __init__(self, hparams):
        super(ShopeeLightning, self).__init__()
        self.hparams = hparams
        self.model = Backbone(hparams)
        self.arc_loss = SphereProduct(hparams.embeddings, hparams.num_classes)
        self.criterion = nn.CrossEntropyLoss()  # FocalLoss(gamma=2)

    def prepare_data(self) -> None:
        lines = read_csv(self.hparams.label_csv)
        lines = np.array(lines)
        lines = lines[np.argsort(lines[:, -1])]
        self.train_lines, self.val_lines = lines[:int(len(lines) * 0.8)], lines[int(len(lines) * 0.8):]
        self.train_dataset = ShopeeDataset(
            self.hparams,
            self.train_lines,
            transform=transforms.Compose([
                transforms.Resize((self.hparams.input_size, self.hparams.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        )

        self.val_dataset = ShopeeDataset(
            self.hparams,
            self.val_lines,
            transform=transforms.Compose([
                transforms.Resize((self.hparams.input_size, self.hparams.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers
        )

    def train_dataloader(self) -> Any:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        fnames, imgs, labels = batch
        features = self.model(imgs)
        logits = self.arc_loss(features, labels)
        loss = self.criterion(logits, labels)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        fnames, imgs, labels = batch
        embeddings = self.model(imgs)
        return {'embeddings': embeddings}

    def validation_epoch_end(self, outputs: List[Any]) -> Any:
        embedding_matrix = []
        gt = self.val_dataset.gt
        for output in outputs:
            output_embeddings = output['embeddings']
            if torch.is_tensor(output_embeddings):
                output_embeddings = output_embeddings.detach().cpu().numpy()
            embedding_matrix.append(output_embeddings)
        embedding_matrix = np.vstack(embedding_matrix)
        sim_matrix = cosine_similarity(embedding_matrix)
        sim_matrix = (sim_matrix - np.min(sim_matrix)) / (np.max(sim_matrix) - np.min(sim_matrix))
        sim_matrix = np.triu(sim_matrix)
        dice_score = dice(sim_matrix > 0.5, gt[:len(sim_matrix), :len(sim_matrix)])
        self.log("val/dice", dice_score, on_step=False, on_epoch=True, prog_bar=True)

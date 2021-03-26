from typing import Union, List, Any

import numpy as np
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.metrics import F1
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from models.meta_model import MetaNet
from utils import read_csv, ShopeeDataset
from utils.loss import SphereProduct


class ShopeeLightning(LightningModule):
    def __init__(self, hparams):
        super(ShopeeLightning, self).__init__()
        self.hparams = hparams
        self.model = MetaNet(hparams)
        # ArcFaceLoss(s=30, m=0.4)
        self.metric_crit = SphereProduct(
            in_features=hparams.text_embedding_size + hparams.image_embedding_size,
            out_features=hparams.num_classes)
        self.ce = CrossEntropyLoss()
        self.f1 = F1(num_classes=hparams.num_classes)

    def prepare_data(self) -> None:
        lines = read_csv(self.hparams.label_csv)
        lines = np.array(lines)
        lines = lines[np.argsort(lines[:, -1])]
        label_map = {label: i for i, label in enumerate(np.unique(np.array(lines)[:, 4]))}
        train_lines, val_lines = lines[:int(len(lines) * 0.8)], lines[int(len(lines) * 0.8):]

        # class_weights = get_class_weights(lines, label_map, self.hparams.num_classes)
        # self.metric_crit.weight = class_weights

        self.train_dataset = ShopeeDataset(
            self.hparams,
            lines=train_lines,
            label_map=label_map,
            transform=transforms.Compose([
                transforms.Resize((self.hparams.input_size, self.hparams.input_size)),
                transforms.ToTensor()
            ])
        )

        self.val_dataset = ShopeeDataset(
            self.hparams,
            lines=val_lines,
            label_map=label_map,
            transform=transforms.Compose([
                transforms.Resize((self.hparams.input_size, self.hparams.input_size)),
                transforms.ToTensor(),
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
        optimizer = torch.optim.Adam(
            [{'params': self.model.parameters()}, {'params': self.metric_crit.parameters()}],
            lr=self.hparams.lr)
        return optimizer

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        fnames, imgs, sentences, labels = batch
        outputs = self.model((imgs, sentences))
        f = self.metric_crit(outputs, labels)
        pred_values, pred_indices = torch.max(f, dim=-1)
        self.f1.update(preds=pred_indices, target=labels)
        loss = self.ce(f, labels)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        return {'loss': loss}

    def training_epoch_end(self, outputs: List[Any]):
        self.log("train/val", self.f1.compute(), on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        fnames, imgs, sentences, labels = batch
        outputs = self.model((imgs, sentences))
        f = self.metric_crit(outputs, labels)
        return {'fnames': fnames, 'embeddings': outputs.detach().cpu().numpy()}

    def validation_epoch_end(self, outputs: List[Any]) -> Any:
        gt = self.val_dataset.gt
        embedding_dic = {}
        for output in outputs:
            for fname, embedding in zip(output['fnames'], output['embeddings']):
                embedding_dic[fname] = embedding

        fnames, embeddings = list(embedding_dic.keys()), list(embedding_dic.values())
        sim_matrix = cosine_similarity(embeddings)
        sim_matrix = (sim_matrix - np.min(sim_matrix)) / (np.max(sim_matrix) - np.min(sim_matrix))

        TP, FP, FN = 0, 0, 0
        for i, fname in enumerate(fnames):
            gt_indices = gt[fname]
            pred_indices = np.where(sim_matrix[i] > 0.5)[0]
            TP += len(set(gt_indices).intersection(set(pred_indices)))
            FP += len(set(pred_indices) - set(gt_indices))
            FN += len(set(gt_indices) - set(pred_indices))

        self.log("val/dice", TP / (TP + 0.5 * (FP + FN)), on_step=False, on_epoch=True, prog_bar=True)

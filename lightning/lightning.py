from typing import Union, List, Any

import albumentations
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import LightningModule
from pytorch_lightning.metrics import F1
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from models.meta_model import MetaNet
from utils import (
    read_csv,
    ShopeeDataset,
    ShopeeTestDataset,
    compute_cosine_similarity,
    write_submission
)
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
            transform=albumentations.Compose([
                albumentations.Resize(self.hparams.input_size, self.hparams.input_size),
                albumentations.HorizontalFlip(p=0.5),
                albumentations.RandomBrightnessContrast(p=0.5, brightness_limit=(-0.2, 0.2),
                                                        contrast_limit=(-0.2, 0.2)),
                albumentations.HueSaturationValue(p=0.5, hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2),
                albumentations.ShiftScaleRotate(p=0.5, shift_limit=0.0625, scale_limit=0.2, rotate_limit=20),
                albumentations.CoarseDropout(p=0.5),
                albumentations.Normalize(),
                ToTensorV2()
            ])
        )

        self.val_dataset = ShopeeDataset(
            self.hparams,
            lines=val_lines,
            label_map=label_map,
            transform=albumentations.Compose([
                albumentations.Resize(self.hparams.input_size, self.hparams.input_size),
                albumentations.Normalize(),
                ToTensorV2()
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

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        test_lines = read_csv(self.hparams.test_csv)
        test_dataset = ShopeeTestDataset(
            self.hparams,
            lines=test_lines,
            transform=albumentations.Compose([
                albumentations.Resize(self.hparams.input_size, self.hparams.input_size),
                albumentations.Normalize()
            ])
        )
        return DataLoader(
            test_dataset,
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

    def training_epoch_end(self, outputs: List[Any]) -> None:
        f1_value = self.f1.compute()
        logs = {"train/f1": f1_value}
        self.logger.log_metrics(logs, step=self.current_epoch)
        self.log("train/f1", f1_value, on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        fnames, imgs, sentences, labels = batch
        outputs = self.model((imgs, sentences))
        return {'fnames': fnames, 'embeddings': outputs.detach().cpu().numpy()}

    def validation_epoch_end(self, outputs: List[Any]) -> Any:
        gt = self.val_dataset.gt
        embedding_dic = {}
        fnames = []
        for output in outputs:
            for fname, embedding in zip(output['fnames'], output['embeddings']):
                embedding_dic[fname] = embedding
                fnames.append(fname)
        sim_matrix = compute_cosine_similarity(embedding_dic)

        TP, FP, FN = 0, 0, 0
        for i, fname in enumerate(fnames):
            gt_indices = gt[fname]
            pred_indices = np.where(sim_matrix[i] > 0.7)[0]
            TP += len(set(gt_indices).intersection(set(pred_indices)))
            FP += len(set(pred_indices) - set(gt_indices))
            FN += len(set(gt_indices) - set(pred_indices))

        f1_value = TP / (TP + 0.5 * (FP + FN))
        logs = {"val/f1": f1_value}
        self.logger.log_metrics(logs, step=self.current_epoch)
        self.log("val/f1", f1_value, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        fnames, imgs, sentences = batch
        outputs = self.model((imgs, sentences))
        return {'fnames': fnames, 'embeddings': outputs.detach().cpu().numpy()}

    def test_epoch_end(self, outputs: List[Any]) -> None:
        embedding_dic = {}
        fnames = []
        for output in outputs:
            for fname, embedding in zip(output['fnames'], output['embeddings']):
                embedding_dic[fname] = embedding
                fnames.append(fname)
        sim_matrix = compute_cosine_similarity(embedding_dic)

        fnames = np.array(fnames)
        submission = {'posting_id': [], 'matches': []}
        for i, fname in enumerate(fnames):
            pred_indices = np.where(sim_matrix[i] > 0.7)[0]
            pred_string = ' '.join(np.unique(fnames[pred_indices]))
            submission['posting_id'].append(fname)
            submission['matches'].append(pred_string)
        write_submission(submission, '/kaggle/working/')

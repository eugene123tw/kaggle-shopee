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
    ShopeeTestDataset,
    compute_cosine_similarity,
    write_submission,
    compute_f1_score
)
from utils.loss import SphereProduct


class ShopeeLightning(LightningModule):
    def __init__(self, hparams):
        super(ShopeeLightning, self).__init__()
        self.hparams = hparams
        self.model = MetaNet(hparams)
        # ArcFaceLoss(s=30, m=0.4)
        self.metric_crit = SphereProduct(
            in_features=hparams.output_feature_size,
            out_features=hparams.num_classes)
        self.ce = CrossEntropyLoss()
        self.f1 = F1(num_classes=hparams.num_classes)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        test_lines = read_csv(self.hparams.test_csv)
        test_dataset = ShopeeTestDataset(
            self.hparams,
            lines=test_lines,
            transform=albumentations.Compose([
                albumentations.Resize(self.hparams.input_size, self.hparams.input_size),
                albumentations.CenterCrop(height=self.hparams.center_crop, width=self.hparams.center_crop),
                albumentations.Normalize(),
                ToTensorV2(),
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
        fnames, imgs, sentences, labels, gt = batch
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
        fnames, imgs, sentences, labels, gt = batch
        outputs = self.model((imgs, sentences))
        return {
            'fnames': fnames,
            'embeddings': outputs.detach().cpu().numpy(),
            'gt': gt,
        }

    def validation_epoch_end(self, outputs: List[Any]) -> Any:
        gt = {}
        fnames, embeddings = [], []
        for output in outputs:
            for fname, embedding, fname_gt in zip(
                    output['fnames'],
                    output['embeddings'],
                    output['gt']
            ):
                gt[fname] = fname_gt
                embeddings.append(embedding)
                fnames.append(fname)

        fnames = np.array(fnames)
        embeddings = np.array(embeddings)

        best_f1 = 0
        best_thres = 0

        for thres in np.arange(0.5, 0.95, 0.05):
            pred_dict = compute_cosine_similarity(embeddings, fnames, batch_compute=True, threshold=thres,
                                                  top_k=self.hparams.top_k)
            f1_value = compute_f1_score(pred_dict, gt)
            if f1_value > best_f1:
                best_thres = thres
                best_f1 = f1_value

        logs = {"val/f1": best_f1, "val/thres": best_thres}
        self.logger.log_metrics(logs, step=self.current_epoch)
        self.log("val/f1", best_f1, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        fnames, imgs, sentences = batch
        outputs = self.model((imgs, sentences))
        return {'fnames': fnames, 'embeddings': outputs.detach().cpu().numpy()}

    def test_epoch_end(self, outputs: List[Any]) -> None:
        fnames, embeddings = [], []
        for output in outputs:
            for fname, embedding, sentence in zip(
                    output['fnames'],
                    output['embeddings'],
            ):
                embeddings.append(embedding)
                fnames.append(fname)

        fnames = np.array(fnames)
        embeddings = np.array(embeddings)
        pred_dict = compute_cosine_similarity(embeddings,
                                              fnames,
                                              threshold=self.hparams.score_threshold,
                                              top_k=self.hparams.top_k,
                                              batch_compute=True)
        result = {}
        for i, fname in enumerate(fnames):
            pred_fnames = pred_dict[fname]
            pred_string = ' '.join(pred_fnames)
            result[fname] = pred_string
        write_submission(result, '/kaggle/working/')

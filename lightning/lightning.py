from typing import List, Any

import cupy as cp
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from pytorch_lightning.metrics import F1
from torch.nn import CrossEntropyLoss

from models.meta_model import MetaNet
from utils import (
    compute_f1_score, compute_cosine_similarity, compute_cosine_similarity_np
)
from utils.loss import SphereProduct, ArcMarginProduct


class ShopeeLightning(LightningModule):
    def __init__(self, hparams):
        super(ShopeeLightning, self).__init__()
        self.hparams = hparams
        self.model = MetaNet(hparams)
        if hparams.d_metric == 'ArcMarginProduct':
            self.metric_crit = ArcMarginProduct(
                in_features=hparams.output_feature_size,
                out_features=hparams.num_classes)
        elif hparams.loss == 'SphereProduct':
            self.metric_crit = SphereProduct(
                in_features=hparams.output_feature_size,
                out_features=hparams.num_classes)
        else:
            raise NotImplemented("UNKNOWN D-METRIC")
        self.ce = CrossEntropyLoss()
        self.f1 = F1(num_classes=hparams.num_classes)

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
        outputs = F.normalize(self.model((imgs, sentences)))

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
            pred_dict = compute_cosine_similarity_np(embeddings=embeddings,
                                                     fnames=fnames,
                                                     threshold=thres,
                                                     top_k=self.hparams.top_k)
            # pred_dict = knn_similarity(
            #     embeddings,
            #     fnames,
            #     n_neighbors=50 if len(fnames) > 3 else len(fnames),
            #     threshold=thres)
            f1_value = compute_f1_score(pred_dict, gt)
            if f1_value > best_f1:
                best_thres = thres
                best_f1 = f1_value

        logs = {"val/f1": best_f1, "val/thres": best_thres}
        self.logger.log_metrics(logs, step=self.current_epoch)
        self.log("val/f1", best_f1, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        fnames, imgs, sentences = batch
        outputs = F.normalize(self.model((imgs, sentences)))
        return {'fnames': fnames, 'embeddings': outputs.detach().cpu().numpy()}

    def test_epoch_end(self, outputs: List[Any]):
        fnames, embeddings = [], []
        for output in outputs:
            for fname, embedding in zip(output['fnames'], output['embeddings']):
                embeddings.append(embedding)
                fnames.append(fname)

        fnames = np.array(fnames)
        embeddings = cp.array(embeddings)

        # result = knn_similarity(
        #     embeddings,
        #     fnames,
        #     n_neighbors=50 if len(fnames) > 3 else len(fnames),
        #     threshold=self.hparams.score_threshold)

        pred_dict = compute_cosine_similarity(embeddings=embeddings,
                                              fnames=fnames,
                                              threshold=self.hparams.score_threshold,
                                              top_k=self.hparams.top_k)

        self.test_results = pred_dict

from typing import Union, List, Any

import albumentations
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import LightningModule
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from models.meta_model import ImageBackbone, SentenceBackbone
from utils import (
    read_csv,
    ShopeeTestDataset,
    compute_cosine_similarity,
    write_submission,
    compute_f1_score
)
from utils.loss import SphereProduct
from torch import nn

class MultiModelShopeeLightning(LightningModule):
    def __init__(self, hparams):
        super(MultiModelShopeeLightning, self).__init__()
        self.hparams = hparams
        self.image_model = ImageBackbone(hparams)
        self.bert_model = SentenceBackbone(hparams)

        self.image_metric_crit = SphereProduct(
            in_features=hparams.image_embedding_size,
            out_features=hparams.num_classes)

        self.bert_metric_crit = nn.Sequential(
            nn.Linear(self.text_embedding_size, hparams.text_embedding_size, bias=False),
            nn.BatchNorm1d(hparams.num_classes)
        )

        self.ce = CrossEntropyLoss()

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        test_lines = read_csv(self.hparams.test_csv)
        test_dataset = ShopeeTestDataset(
            self.hparams,
            lines=test_lines,
            transform=albumentations.Compose([
                albumentations.Resize(self.hparams.input_size, self.hparams.input_size),
                albumentations.Normalize(),
                ToTensorV2()
            ])
        )
        return DataLoader(
            test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers
        )

    def configure_optimizers(self):
        img_optimizer = torch.optim.Adam(
            [{'params': self.image_model.parameters()}, {'params': self.image_metric_crit.parameters()}],
            lr=self.hparams.lr)

        bert_optimizer = torch.optim.Adam(
            [{'params': self.bert_model.parameters()}, {'params': self.bert_metric_crit.parameters()}],
            lr=self.hparams.lr)
        return [img_optimizer, bert_optimizer]

    def training_step(self, batch, batch_idx, optimizer_idx):
        fnames, imgs, sentences, labels, gt = batch

        if optimizer_idx == 0:
            img_f = self.image_metric_crit(self.image_model(imgs), labels)
            img_loss = self.ce(img_f, labels)
            self.log("train/img_loss", img_loss, on_step=True, on_epoch=False, prog_bar=True)
            return {'loss': img_loss}

        bert_f = self.bert_metric_crit(self.bert_model(sentences), labels)
        bert_loss = self.ce(bert_f, labels)
        self.log("train/bert_loss", bert_loss, on_step=True, on_epoch=False, prog_bar=True)
        return {'loss': bert_loss}

    def validation_step(self, batch, batch_idx):
        fnames, imgs, sentences, labels, gt = batch
        img_outputs = self.image_model(imgs)
        bert_outputs = self.bert_model(sentences)
        return {'fnames': fnames,
                'img_embeddings': img_outputs.detach().cpu().numpy(),
                'bert_embeddings': bert_outputs.detach().cpu().numpy(),
                'gt': gt}

    def validation_epoch_end(self, outputs: List[Any]) -> Any:
        gt = {}
        fnames, img_embeddings, bert_embeddings = [], [], []
        
        for output in outputs:
            for fname, img_embed, bert_embed, fname_gt in zip(
                output['fnames'], 
                output['img_embeddings'], 
                output['bert_embeddings'], 
                output['gt']
                ):
                gt[fname] = fname_gt
                img_embeddings.append(img_embed)
                bert_embeddings.append(bert_embed)
                fnames.append(fname)
        del outputs

        fnames = np.array(fnames)
        img_embeddings = np.array(img_embeddings)
        bert_embeddings = np.array(bert_embeddings)
        img_pred_dict = compute_cosine_similarity(img_embeddings,
                                              fnames,
                                              threshold=self.hparams.score_threshold,
                                              top_k=self.hparams.top_k,
                                              batch_compute=True)
        del img_embeddings

        bert_pred_dict = compute_cosine_similarity(bert_embeddings,
                                              fnames,
                                              threshold=self.hparams.score_threshold,
                                              top_k=self.hparams.top_k,
                                              batch_compute=True)
        del bert_embeddings
        
        pred_dict = {}
        for fname in fnames:
            pred = np.intersect1d(img_pred_dict[fname], bert_pred_dict[fname])
            pred_dict[fname] = pred
        f1_score = compute_f1_score(pred_dict, gt)
        
        logs = {"val/f1": f1_score}
        self.logger.log_metrics(logs, step=self.current_epoch)
        self.log("val/f1", f1_score, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        fnames, imgs, sentences = batch
        outputs = self.model((imgs, sentences))
        return {'fnames': fnames, 'embeddings': outputs.detach().cpu().numpy()}

    def test_epoch_end(self, outputs: List[Any]) -> None:
        fnames, embeddings = [], []
        for output in outputs:
            for fname, embedding in zip(output['fnames'], output['embeddings']):
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

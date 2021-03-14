import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.metrics.classification import Fbeta
from torch.nn import Softmax

from models.meta_model import Backbone
from thirdparty.arc_face.models import FocalLoss, ArcMarginProduct


class ShopeeLightning(LightningModule):
    def __init__(self, hparams):
        super(ShopeeLightning, self).__init__()
        self.hparams = hparams
        self.model = Backbone(hparams)
        self.metric = ArcMarginProduct(hparams.embeddings, hparams.num_classes, easy_margin=False)
        self.criterion = FocalLoss(gamma=2)
        self.f_measure = Fbeta(num_classes=hparams.num_classes)
        self.softmax = Softmax(dim=1)

    def configure_optimizers(self):
        # TODO: make sure all weights are included (the weight in ArcLoss especially)
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()}, {'params': self.metric.parameters()}
        ], lr=self.hparams.lr)
        return optimizer

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        features = self.model(imgs)
        features = self.metric(features, labels)
        loss = self.criterion(features, labels)
        self.log_dict({'train_loss': loss})
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        features = self.model(imgs)
        features = self.metric(features, labels)
        val_loss = self.criterion(features, labels)
        p = self.softmax(features)
        pred_score, pred_indices = torch.max(p, dim=-1)
        self.f_measure.update(pred_indices, labels)
        metrics = {'val_loss': val_loss, 'val_accu': self.f_measure.compute()}
        self.log_dict(metrics)
        return metrics

    def validation_epoch_end(self, outputs):
        return {'val_accuracy': self.f_measure.compute()}

import torch
import torch.nn.functional as F
from torch import nn

import timm
from models import resnet_backbone
from utils.loss import ArcMarginProduct


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, p_trainable=True):
        super(GeM, self).__init__()
        if p_trainable:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.p = p
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(
            self.eps) + ')'


class Backbone(nn.Module):
    def __init__(self, hparams):
        super(Backbone, self).__init__()
        self.hparams = hparams

        if 'resnet' in hparams.backbone:
            self.backbone = resnet_backbone[self.hparams.backbone](pretrained=self.hparams.pretrained,
                                                                   num_classes=hparams.embeddings)
        elif 'efficientnet' in hparams.backbone:
            self.backbone = timm.create_model(hparams.backbone, pretrained=self.hparams.pretrained)
            self.out_features = self.backbone.classifier.in_features

    def forward(self, x):
        x = self.backbone.forward_features(x)
        return x


class MetaNet(nn.Module):
    def __init__(self, hparams):
        super(MetaNet, self).__init__()
        self.hparams = hparams
        self.backbone = Backbone(hparams)
        self.global_pool = GeM(p_trainable=hparams.p_trainable)
        self.embedding_size = hparams.embedding_size

        if hparams.neck == "option-D":
            self.neck = nn.Sequential(
                nn.Linear(self.backbone.out_features, self.embedding_size, bias=True),
                nn.BatchNorm1d(self.embedding_size),
                torch.nn.PReLU()
            )
        elif hparams.neck == "option-F":
            self.neck = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(self.backbone.out_features, self.embedding_size, bias=True),
                nn.BatchNorm1d(self.embedding_size),
                torch.nn.PReLU()
            )
        else:
            self.neck = nn.Sequential(
                nn.Linear(self.backbone.out_features, self.embedding_size, bias=False),
                nn.BatchNorm1d(self.embedding_size),
            )

        self.head = ArcMarginProduct(self.embedding_size, hparams.num_classes)

    def forward(self, x, get_embeddings=False):

        x = self.backbone(x)

        x = self.global_pool(x)
        x = x[:, :, 0, 0]

        x = self.neck(x)

        logits = self.head(x)

        if get_embeddings:
            return {'logits': logits, 'embeddings': x}
        else:
            return {'logits': logits}

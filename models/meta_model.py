from torch import nn

from models import resnet_backbone


class Backbone(nn.Module):
    def __init__(self, hparams):
        super(Backbone, self).__init__()
        self.hparams = hparams
        self.backbone = resnet_backbone[self.hparams.backbone](pretrained=self.hparams.pretrained,
                                                               num_classes=hparams.embeddings)

    def forward(self, x):
        x = self.backbone(x)
        return x

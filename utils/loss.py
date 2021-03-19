import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, m=0.4):
        super(ArcMarginProduct, self).__init__()
        self.out_features = out_features
        self.m = m
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, labels):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        x_norm = F.normalize(input)
        W_norm = F.normalize(self.weight, dim=0)
        cosine = x_norm @ W_norm
        cosine = cosine.clip(-1 + 1e-7, 1 - 1e-7)
        arcosine = cosine.arccos()
        arcosine += F.one_hot(labels, num_classes=self.out_features) * self.m
        cosine2 = arcosine.cos()
        loss = F.cross_entropy(cosine2, labels)
        return loss

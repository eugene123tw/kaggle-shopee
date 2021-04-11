import timm
import torch
import torch.nn.functional as F
from torch import nn
from transformers import DistilBertTokenizer, DistilBertModel


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


class ImageBackbone(nn.Module):
    def __init__(self, hparams):
        super(ImageBackbone, self).__init__()
        self.hparams = hparams

        self.backbone = timm.create_model(hparams.backbone, pretrained=self.hparams.pretrained)

        if 'resnet' in hparams.backbone:
            self.out_features = self.backbone.fc.in_features
        elif 'efficientnet':
            self.out_features = self.backbone.classifier.in_features
        else:
            raise NotImplemented

        # Define Last Pooling Layer
        self.global_pool = GeM(p_trainable=hparams.p_trainable)

        # Define Neck Layer
        if hparams.neck == "option-D":
            self.neck = nn.Sequential(
                nn.Linear(self.out_features, hparams.image_embedding_size, bias=True),
                nn.BatchNorm1d(hparams.image_embedding_size),
                torch.nn.PReLU()
            )
        elif hparams.neck == "option-F":
            self.neck = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(self.out_features, hparams.image_embedding_size, bias=True),
                nn.BatchNorm1d(hparams.image_embedding_size),
                torch.nn.PReLU()
            )
        else:
            self.neck = nn.Sequential(
                nn.Linear(self.out_features, hparams.image_embedding_size, bias=True),
                nn.BatchNorm1d(hparams.image_embedding_size),
            )

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.global_pool(x)
        x = x[:, :, 0, 0]
        x = self.neck(x)
        return x


class SentenceBackbone(nn.Module):
    def __init__(self, hparams):
        super(SentenceBackbone, self).__init__()
        self.hparams = hparams
        self.tokenizer = DistilBertTokenizer.from_pretrained(hparams.text_backbone)
        self.text_backbone = DistilBertModel.from_pretrained(hparams.text_backbone)

    def forward(self, sentence):
        tokens_output = self.tokenizer(
            list(sentence),
            return_tensors="pt",
            padding=True,
            return_length=False,
            return_token_type_ids=False,
            return_attention_mask=True,
            truncation="only_first",
            max_length=self.hparams.sentence_max_length,
        )
        tokens, attention_mask = tokens_output["input_ids"], tokens_output["attention_mask"]

        output = self.text_backbone(tokens.cuda(), attention_mask.cuda())
        word_embeddings = output.last_hidden_state
        word_embeddings = word_embeddings[:, 0, :]

        # obtaining CLS token state which is the first token.
        return word_embeddings


class MetaNet(nn.Module):
    def __init__(self, hparams):
        super(MetaNet, self).__init__()
        self.hparams = hparams
        self.image_backbone = ImageBackbone(hparams)
        self.sentence_backbone = SentenceBackbone(hparams)
        self.embedding_size = hparams.image_embedding_size + hparams.text_embedding_size
        self.norm = nn.BatchNorm1d(self.embedding_size)

    def forward(self, input):
        img, sentence = input
        img_embed = self.image_backbone(img)
        sentence_embed = self.sentence_backbone(sentence)

        x = torch.cat((img_embed, sentence_embed), -1)
        x = self.norm(x)
        return x

from typing import Any

from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback


class FeatureEmbeddingCache(Callback):
    def __init__(self, hparams):
        self.embeddings = {}

    def _extract_embeddings(self, outputs):
        embeddings = outputs[0][0].extra['embeddings'].detach().cpu().numpy()
        fnames = outputs[0][0].extra['fnames']
        for fname, embedding in zip(fnames, embeddings):
            self.embeddings[fname] = embedding

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self._extract_embeddings(outputs)

    def on_validation_batch_end(
            self, trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        self._extract_embeddings(outputs)

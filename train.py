from typing import Dict

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from sklearn.feature_extraction.text import TfidfVectorizer

from lightning import *
from utils import compute_cosine_similarity, combine_pred_dicts, write_submission


def train(config: DictConfig):
    lightning = ShopeeLightning(config)
    data_module = ShopeeTrainValDataModule(config)

    checkpoint_callback = ModelCheckpoint(
        dirpath=config.model_checkpoint.dirpath,
        filename=config.model_checkpoint.filename,
        monitor=config.model_checkpoint.monitor,
        mode=config.model_checkpoint.mode,
        save_last=config.model_checkpoint.save_last,
        save_top_k=config.model_checkpoint.save_top_k
    )

    stopping_callback = EarlyStopping(
        monitor=config.early_stopping.monitor,
        patience=config.early_stopping.patience,
        mode=config.early_stopping.mode,
        min_delta=config.early_stopping.min_delta,
    )

    logger = WandbLogger(
        project=config.wandb.project,
        name=config.wandb.name,
        log_model=True
    )
    trainer = Trainer(
        gpus=config.gpus,
        max_epochs=config.epochs,
        logger=logger,
        callbacks=[
            checkpoint_callback,
            stopping_callback
        ],
    )
    trainer.fit(lightning, data_module)


def tfidf(config, test_dm) -> Dict:
    model = TfidfVectorizer(stop_words='english', binary=True, max_features=25_000)

    test_dm.setup('test')
    fnames, sentences = [], []
    for batch in test_dm.test_dataloader():
        fnames.extend(batch[0])
        sentences.extend(batch[2])
    text_embeddings = model.fit_transform(sentences).toarray()
    pred_dict = compute_cosine_similarity(text_embeddings,
                                          np.array(fnames),
                                          threshold=config.score_threshold,
                                          top_k=config.top_k,
                                          batch_compute=True)
    return pred_dict


def test(config: DictConfig):
    # config.update(
    #     {
    #         'weights': "/home/yuchunli/git/kaggle-shopee/logs/runs/2021-04-05/21-25-16/checkpoints/epoch=9-val/f1=0.866.ckpt",
    #         'text_backbone': '/home/yuchunli/_MODELS/huggingface/distilbert-base-indonesian',
    #         'pretrained': False
    #     }
    # )

    checkpoint = torch.load(config.weights)
    lightning = ShopeeLightning(config)
    lightning.load_state_dict(checkpoint['state_dict'])
    test_dm = ShopeeTestDataModule(config)
    trainer = Trainer(gpus=config.gpus)
    trainer.test(lightning, datamodule=test_dm)
    dnn_result = lightning.test_results

    tfide_result = tfidf(config, test_dm)
    result = combine_pred_dicts([dnn_result, tfide_result])
    write_submission(result, '/kaggle/working/')


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    if not config.testing:
        return train(config)
    else:
        return test(config)


if __name__ == "__main__":
    main()

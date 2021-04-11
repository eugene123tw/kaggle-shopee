from typing import Dict

import cudf
import cupy as cp
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from cuml.feature_extraction.text import TfidfVectorizer
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from lightning import *
from utils import (
    knn_similarity,
    compute_cosine_similarity,
    compute_f1_score,
    combine_pred_dicts
)
from utils import write_submission


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
        num_sanity_val_steps=-1,
        logger=logger,
        callbacks=[
            checkpoint_callback,
            stopping_callback
        ],
    )
    trainer.fit(lightning, data_module)


def tfidf(config, test_dm) -> Dict:
    model = TfidfVectorizer(stop_words='english', binary=True, max_features=25000)

    test_dm.setup()
    fnames, sentences = [], []
    for batch in test_dm.test_dataloader():
        fnames.extend(batch[0])
        sentences.extend(batch[2])
    text_embeddings = model.fit_transform(cudf.Series(sentences)).toarray()
    pred_dict = compute_cosine_similarity(embeddings=text_embeddings,
                                          fnames=np.array(fnames),
                                          threshold=config.score_threshold,
                                          top_k=config.top_k)
    return pred_dict


def validate(config: DictConfig):
    config.update(
        {
            'weights': "/home/yuchunli/git/kaggle-shopee/logs/runs/2021-04-09/12-29-29/checkpoints/epoch=9-val/f1=0.877.ckpt",
            'text_backbone': '/home/yuchunli/_MODELS/huggingface/distilbert-base-indonesian',
            'pretrained': False
        }
    )

    checkpoint = torch.load(config.weights)
    lightning = ShopeeLightning(config)
    lightning.load_state_dict(checkpoint['state_dict'])
    train_dm = ShopeeTrainValDataModule(config)
    train_dm.setup()
    fnames, embeddings = [], []
    gt = {}
    lightning.eval()
    lightning.cuda()
    with torch.no_grad():
        for batch in train_dm.val_dataloader():
            b_fname, imgs, sentences, labels, b_gt = batch
            embedding = F.normalize(lightning((imgs.cuda(), sentences))).detach().cpu().numpy()
            for fname, gt_list, embed in zip(b_fname, b_gt, embedding):
                gt[fname] = gt_list
                fnames.append(fname)
                embeddings.append(embed)

    fnames = np.array(fnames)
    embeddings = cp.array(embeddings)
    pred_dict = knn_similarity(
        embeddings,
        fnames,
        n_neighbors=50 if len(fnames) > 3 else len(fnames),
        threshold=0.9)
    f1_value = compute_f1_score(pred_dict, gt)
    print(f1_value)


def test(config: DictConfig):
    # config.update(
    #     {
    #         'weights': "/home/yuchunli/git/kaggle-shopee/logs/runs/2021-04-11/19-54-22/checkpoints/epoch=9-val/f1=0.762.ckpt",
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
    result = lightning.test_results
    if config.tfidf:
        tfidf_result = tfidf(config, test_dm)
        result = combine_pred_dicts([result, tfidf_result], method='union')
    write_submission(result, '/kaggle/working/')


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    # validate(config)
    if not config.testing:
        return train(config)
    else:
        return test(config)


if __name__ == "__main__":
    main()

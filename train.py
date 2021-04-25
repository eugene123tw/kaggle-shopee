import glob
from typing import *

import cudf
import cupy as cp
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from cuml.feature_extraction.text import TfidfVectorizer
from hydra.experimental import compose, initialize
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from lightning import *
from utils import (
    compute_cosine_similarity,
    compute_f1_score,
    combine_pred_dicts,
    ensemble_prob_dicts
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

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

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
            stopping_callback,
            lr_monitor
        ],
    )
    trainer.fit(lightning, data_module)


def tfidf(config) -> Tuple[Dict, Dict]:
    model = TfidfVectorizer(stop_words='english', binary=True, max_features=25000)
    test_dm = ShopeeTestDataModule(config)
    test_dm.setup()
    fnames, sentences = [], []
    for batch in test_dm.test_dataloader():
        fnames.extend(batch[0])
        sentences.extend(batch[2])
    text_embeddings = model.fit_transform(cudf.Series(sentences)).toarray()
    pred_dict, prob_dict = compute_cosine_similarity(embeddings=text_embeddings,
                                                     fnames=np.array(fnames),
                                                     threshold=config.score_threshold,
                                                     top_k=config.top_k,
                                                     get_prob=config.prob_ensemble)
    return pred_dict, prob_dict


def bert(config: DictConfig):
    config.update(
        {
            'text_backbone': '/home/yuchunli/_MODELS/huggingface/distilbert-base-indonesian',
            'pretrained': False
        }
    )
    lightning = ShopeeLightning(config)
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
    best_f1 = 0
    best_thres = 0
    for thres in np.arange(0, 0.95, 0.05):
        pred_dict, prob_dict = compute_cosine_similarity(embeddings=embeddings,
                                                         fnames=fnames,
                                                         threshold=thres,
                                                         top_k=config.top_k,
                                                         get_prob=config.prob_ensemble)
        f1_value = compute_f1_score(pred_dict, gt)
        if f1_value > best_f1:
            best_thres = thres
            best_f1 = f1_value
    print(best_thres)


def validate(config: DictConfig):
    # config.update(
    #     {
    #         'weights': "/home/yuchunli/git/kaggle-shopee/logs/runs/2021-04-09/12-29-29/checkpoints/epoch=9-val/f1=0.877.ckpt",
    #         'text_backbone': '/home/yuchunli/_MODELS/huggingface/distilbert-base-indonesian',
    #         'pretrained': False
    #     }
    # )

    # checkpoint = torch.load(config.weights)
    lightning = ShopeeLightning(config)
    # lightning.load_state_dict(checkpoint['state_dict'])
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
    best_thres = 0
    best_f1 = 0
    for thres in np.arange(0, 0.95, 0.05):
        pred_dict, prob_dict = compute_cosine_similarity(embeddings=embeddings,
                                                         fnames=fnames,
                                                         threshold=thres,
                                                         top_k=config.top_k,
                                                         get_prob=config.prob_ensemble)
        f1_value = compute_f1_score(pred_dict, gt)
        if f1_value > best_f1:
            best_thres = thres
            best_f1 = f1_value
    print(f"best_f1: {best_f1} best_thres: {best_thres}")


def test(config: DictConfig) -> Tuple[Dict, Dict]:
    checkpoint = torch.load(config.weights)
    lightning = ShopeeLightning(config)
    lightning.load_state_dict(checkpoint['state_dict'])
    test_dm = ShopeeTestDataModule(config)
    trainer = Trainer(gpus=config.gpus)
    trainer.test(lightning, datamodule=test_dm)
    return lightning.test_results, lightning.test_prob_dict


def ensemble(checkpoint_paths: List[str]):
    results = []
    prob_dicts = []
    for ckpt_folder in checkpoint_paths:
        with initialize(config_path=ckpt_folder, job_name="ensemble_app"):
            config = compose(config_name="config.yaml", overrides=[
                "prob_ensemble=true",
            ])
            ckpt_path = glob.glob(ckpt_folder + "/*.ckpt")[0]
            config.weights = ckpt_path
            result, prob_dict = test(config)
            prob_dicts.append(prob_dict)
            results.append(result)

    result, prob_dict = tfidf(config)
    results.append(result)
    prob_dicts.append(prob_dict)

    if config.prob_ensemble:
        print("Combine Prob Dictionaries")
        result = ensemble_prob_dicts(prob_dicts, threshold=0.7)
    else:
        print("Individual Ensemble")
        result = combine_pred_dicts(results, method='union')
    write_submission(result, '/kaggle/working/')


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    # config.update(
    #     {
    #         'weights': "/home/yuchunli/git/kaggle-shopee/logs/runs/2021-04-18/21-51-07/2021-04-18_21-51-07__0.778/f1=0.778.ckpt",
    #         'text_backbone': '/home/yuchunli/_MODELS/huggingface/distilbert-base-indonesian',
    #         'pretrained': False,
    #         'testing': True
    #     }
    # )

    # validate(config)
    # bert(config)
    if not config.testing:
        return train(config)
    else:
        return test(config)


if __name__ == "__main__":
    # ensemble(checkpoint_paths=[
    #     "logs/runs/2021-04-18/21-51-07/2021-04-18_21-51-07__0.778",
    #     "logs/runs/2021-04-24/22-05-05/checkpoints/epoch=4-val"
    # ])
    main()

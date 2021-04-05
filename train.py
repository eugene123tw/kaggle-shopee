import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from lightning import *


def train(config: DictConfig):
    lightning = ShopeeLightning(config)
    # lightning = MultiModelShopeeLightning(config)
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


def test(config: DictConfig):
    config.update(
        {
            'weights': "/home/yuchunli/git/kaggle-shopee/logs/runs/2021-04-03/15-05-32/checkpoints/epoch=1-val/f1=0.700.ckpt",
            'text_backbone': '/home/yuchunli/_MODELS/huggingface/distilbert-base-indonesian',
            'pretrained': False
        }
    )
    checkpoint = torch.load(config.weights)
    lightning = ShopeeLightning(config)
    lightning.load_state_dict(checkpoint['state_dict'])
    trainer = Trainer(gpus=config.gpus)
    trainer.test(lightning)


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    if not config.testing:
        return train(config)
    else:
        return test(config)


if __name__ == "__main__":
    main()

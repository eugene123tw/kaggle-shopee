import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from lightning import *


def train(config: DictConfig):
    lightning = ShopeeLightning(config)
    data_module = ShopeeDataModule(config)
    logger = WandbLogger(save_dir=config.wandb.save_dir, offline=config.wandb.offline)
    trainer = Trainer(
        gpus=config.gpus,
        max_epochs=config.epochs,
        logger=logger
    )
    trainer.fit(lightning, data_module)


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    return train(config)


if __name__ == "__main__":
    main()

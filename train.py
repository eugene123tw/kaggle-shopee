import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from lightning import *


def train(config: DictConfig):
    lightning = ShopeeLightning(config)
    data_module = ShopeeDataModule(config)
    trainer = Trainer(gpus=config.gpus, max_epochs=config.epochs)
    trainer.fit(lightning, data_module)


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    return train(config)


if __name__ == "__main__":
    main()

import hydra
import torch
from omegaconf import DictConfig

from lightning import *


def test(config):
    config.update(
        {
            'weights': "/home/yuchunli/git/kaggle-shopee/logs/runs/2021-04-05/21-25-16/checkpoints/epoch=9-val/f1=0.866.ckpt",
            'text_backbone': '/home/yuchunli/_MODELS/huggingface/distilbert-base-indonesian',
            'pretrained': False
        }
    )
    checkpoint = torch.load(config.weights)
    lightning = ShopeeLightning(config)
    lightning.load_state_dict(checkpoint['state_dict'])

    data_module = ShopeeTestDataModule(config)
    data_module.setup('test')
    for batch in data_module.test_dataloader():
        print(batch)


@hydra.main(config_path="../configs/", config_name="config_kaggle.yaml")
def main(config: DictConfig):
    return test(config)

if __name__ == '__main__':
    main()
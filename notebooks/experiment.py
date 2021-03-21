import matplotlib.pylab as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningModule, LightningDataModule, Trainer
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import F1
from torchvision import datasets, transforms

from utils.loss import ArcMarginProduct, SphereProduct


class Net(nn.Module):

    def __init__(self, num_classes):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 3),
            nn.BatchNorm2d(6),
            nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU())

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class DataModule(LightningDataModule):
    def __init__(self):
        super().__init__()

    def prepare_data(self):
        transform = transforms.Compose([transforms.ToTensor()])
        self.train_dataset = datasets.MNIST('/home/user/_DATASET/torch', train=True, download=True, transform=transform)
        self.val_dataset = datasets.MNIST('/home/user/_DATASET/torch', train=False, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=64, num_workers=4, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=48, num_workers=4)


class TestDataModule(LightningDataModule):
    def __init__(self):
        super().__init__()

    def prepare_data(self):
        transform = transforms.Compose([transforms.ToTensor()])
        self.val_dataset = datasets.MNIST('/home/user/_DATASET/torch', train=False, transform=transform)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=48, num_workers=4)


def plot_embs(embs, ys, ax):
    # ax.axis('off')
    for k in range(10):
        e = embs[ys == k].cpu()
        ax.scatter(e[:, 0], e[:, 1], e[:, 2], s=4, alpha=.2)


class LightningModel(LightningModule):
    def __init__(self, arc_loss):
        super().__init__()
        self.backbone = Net(num_classes=3)
        self.arc_loss = arc_loss
        self.loss = nn.CrossEntropyLoss()
        self.f1 = F1(num_classes=10)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.backbone(x)
        features = self.arc_loss(logits, y)
        loss = self.loss(features, y)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.backbone(x)
        features = self.arc_loss(logits, y)
        pred_probs, pred_indices = torch.max(features, 1)
        self.f1.update(pred_indices, y)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.backbone(x)
        return logits, y

    def test_epoch_end(self, outputs):
        embeddings = torch.empty((0, 3), device='cuda')
        y = torch.empty((0), device='cuda')
        for output in outputs:
            embedding, labels = output
            embeddings = torch.vstack((embeddings, embedding))
            y = torch.hstack((y, labels))

        embeddings = embeddings / embeddings.norm(p=2, dim=1)[:, None]
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        plot_embs(embeddings, y, ax)
        plt.show()

    def validation_epoch_end(self, batch_parts):
        self.log('val/f1', self.f1.compute(), on_step=False, on_epoch=True, prog_bar=True)


if __name__ == "__main__":
    lightning = LightningModel(arc_loss=SphereProduct(3, 10))
    data_module = DataModule()
    trainer = Trainer(max_epochs=10, gpus=1)
    trainer.fit(lightning, data_module)
    trainer.test(datamodule=TestDataModule())

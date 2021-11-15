from Dataloader.Dataloader import LoadFileParameter, SaveFileParameter, DataGenerator, DataModule, WSIQuery
import pytorch_lightning as pl
import sys
import torch
from torch.optim import Adam
import torch.nn as nn
from torchmetrics.functional import accuracy
from torchvision import datasets, models, transforms

class Classifier2D(pl.LightningModule):
    def __init__(self, num_classes=2, backbone=models.densenet121())
        super().__init__()
        self.num_classes = num_classes
        self.backbone = backbone
        self.loss_fcn = nn.CrossEntropyLoss()
        self.model = nn.Sequential(
            self.backbone,
            nn.LazyLinear(512),
            nn.LazyLinear(num_classes)
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, train_batch, batch_idx):
        image, labels = train_batch
        logits = self(image)
        loss = self.loss_fcn(logits, labels)
        return loss

    def validation_step(self, val_batch, batch_idx):
        image, labels = val_batch
        logits = self(image)
        loss = self.loss_fcn(logits, labels)
        return loss

    def testing_step(self, test_batch, batch_idx):
        image, labels = test_batch
        logits = self(image)
        loss = self.loss_fcn(logits, labels)
        return loss

    def predict_step(self, batch):
        image = batch
        return self(image)

    def configure_optimizers(self):
	optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
	return [optimizer], [scheduler]



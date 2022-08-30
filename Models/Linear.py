import pytorch_lightning as pl
import matplotlib.pyplot as plt
from pytorch_lightning import LightningDataModule, LightningModule
import numpy as np
import torch
from torch import nn
import torchvision
from torchvision import datasets, models, transforms
from torchvision import transforms
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import LightningDataModule, LightningModule, Trainer,seed_everything
import sys
import torchio as tio
import sklearn
from pytorch_lightning import loggers as pl_loggers
import torchmetrics

## Model
class Linear(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.LazyLinear(64),
            nn.LayerNorm(64),
            nn.ReLU()
        )
        self.loss_fcn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x.float())

    def training_step(self, batch,batch_idx):
        image,label = batch
        prediction  = self.forward(image)
        loss = self.loss_fcn(prediction, label)
        return loss

    def validation_step(self, batch,batch_idx):
        image,label = batch
        prediction  = self.forward(image)
        loss = self.loss_fcn(prediction, label)
        return loss

    def weights_init(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.LayerNorm):
            nn.init.xavier_uniform_(m.weight.data)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return [optimizer], [scheduler]


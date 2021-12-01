import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from pytorch_lightning import LightningDataModule, LightningModule
import numpy as np
import torch
from torch import nn
from collections import Counter
import torchvision
from torchvision import datasets, models, transforms
from torchvision import transforms
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import LightningDataModule, LightningModule, Trainer,seed_everything
from torchsummary import summary
import sys
import torchio as tio
import sklearn
from pytorch_lightning import loggers as pl_loggers
import torchmetrics

## Models
from Models.Linear2D import Linear2D
from Models.Classifier2D import Classifier2D
from Models.Classifier3D import Classifier3D

class MixModel(LightningModule):
    def __init__(self, module_list):
        super().__init__()
        self.model = module_list
        self.classifier = nn.Sequential(
            nn.LazyLinear(128),
            nn.LazyLinear(1)
        )
        self.accuracy = torchmetrics.AUC(reorder=True)
        self.loss_fcn = torch.nn.BCEWithLogitsLoss()

    def forward(self, x):        
        features =  torch.cat([model(data).detach() for data, model in zip(x, self.model)], dim=1) ## Detach for no grad
        return self.classifier(features)

    def training_step(self, batch,batch_idx):    
        prediction  = self.forward(batch[:-1])
        print(prediction.squeeze(), batch[-1])
        print(prediction.squeeze(dim=1).size(), batch[-1].size())
        loss = self.loss_fcn(prediction.squeeze(dim=1), batch[-1])
        self.log("loss", loss)
        return loss

    def validation_step(self, batch,batch_idx):
        prediction  = self.forward(batch[:-1])
        print(prediction, batch[-1])
        print(prediction.squeeze(dim=1).size(), batch[-1].size())
        loss = self.loss_fcn(prediction.squeeze(dim=1), batch[-1])
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return [optimizer], [scheduler]


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
import sys
import torchio as tio
import sklearn
from pytorch_lightning import loggers as pl_loggers
import torchmetrics

## Models
from Models.Linear import Linear
from Models.Classifier2D import Classifier2D
from Models.Classifier3D import Classifier3D

class MixModel(LightningModule):
    def __init__(self, module_dict, loss_fcn = torch.nn.BCEWithLogitsLoss() ):
        super().__init__()
        self.module_dict = module_dict
        self.classifier = nn.Sequential(
            nn.LazyLinear(128),
            nn.LazyLinear(1)
        )
        self.accuracy = torchmetrics.AUC(reorder=True)
        self.loss_fcn = torch.nn.MSELoss() # loss_fcn

    def forward(self, datadict):
        features = torch.cat([self.module_dict[key](datadict[key]) for key in self.module_dict.keys()], dim=1)
        #for key in self.module_dict.keys():
        #    data = self.module_dict[key](datadict[key])
        #    print(key,data)
        #print("features",features, features.shape)
        return self.classifier(features)

    def training_step(self, batch, batch_idx):
        datadict, label = batch
        prediction  = self.forward(datadict)
        print(prediction, label)
        loss = self.loss_fcn(prediction.squeeze(dim=1), batch[-1])
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        datadict, label = batch
        prediction  = self.forward(datadict)
        val_loss = self.loss_fcn(prediction.squeeze(dim=1), batch[-1])
        print('val_prediction:', prediction, label)
        print('val_loss:', val_loss)
        self.log("val_loss", val_loss)
        return val_loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return [optimizer], [scheduler]


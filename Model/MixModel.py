import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from pytorch_lightning import LightningDataModule, LightningModule
import numpy as np
import torch
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
from Model.Linear2D import Linear2D
from Model.Classifier2D import Classifier2D
from Model.Classifier3D import Classifier3D

class MixModel(LightningModule):
    def __init__(self, module_list:list[nn.Module]):
        super().__init__()
        self.model = module_list
        self.classifier = nn.Sequential(
            nn.LazyLinear(128),
            nn.LazyLinear(2)
        )
        #summary(self.model.to('cuda'), (2,160,160,40))
        self.accuracy = torchmetrics.AUC(reorder=True)
        self.loss_fcn = torch.nn.BCEWithLogitsLoss()

    def forward(self, x):        
        features =  torch.cat([model(data).detach() for data, model in zip(x, self.model)], dim=1) ## Detach for no grad
        return self.classifier(features)

    def training_step(self, batch,batch_idx):
        image,label = batch
        prediction  = self.forward(image)
        loss = self.loss_fcn(prediction.squeeze(), label)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch,batch_idx):
        image,label = batch
        prediction  = self.forward(image)
        loss = self.loss_fcn(prediction.squeeze(), label)
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return [optimizer], [scheduler]


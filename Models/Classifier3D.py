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
import sys
import torchio as tio
from Models.unet3d import UNet3D
import sklearn
from pytorch_lightning import loggers as pl_loggers
import torchmetrics

## Model
class Classifier3D(LightningModule):
    def __init__(self, n_classes = 1, wf=5, depth=3):
        super().__init__()
        self.unet_model = UNet3D(in_channels=1, n_classes = n_classes, depth=depth,wf=wf)
        self.model      = torch.nn.Sequential(
            self.unet_model.encoder,
            torch.nn.Flatten(),
            #torch.nn.MaxPool3d((10,10,1)),
            #torch.nn.LazyLinear(128),
            #torch.nn.LazyLinear(n_classes)            
        )
        self.accuracy = torchmetrics.AUC(reorder=True)
        self.loss_fcn = torch.nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch,batch_idx):
        image,label = batch
        prediction  = self.forward(image)
        loss = self.loss_fcn(prediction.squeeze(), label)
        return {"loss":loss,"prediction":prediction.squeeze(),"label":label}

    def validation_step(self, batch,batch_idx):
        image,label = batch
        prediction  = self.forward(image)
        loss = self.loss_fcn(prediction.squeeze(), label)
        return {"loss":loss,"prediction":prediction.squeeze(),"label":label}
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        return [optimizer], [scheduler]


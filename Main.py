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

## Module - Dataloaders
from Dataloader.Dataloader import DataModule, DataGenerator, LoadSortDataLabel

## Module - Models
from Model.Classifier2D import Classifier2D
from Model.Classifier3D import Classifier3D
from Model.Linear2D import Linear2D
from Model.MixModel import MixModel
## Main
train_transform = tio.Compose([
    tio.RandomAffine(),
    # tio.RescaleIntensity(out_min_max=(0, 1))
])

val_transform = tio.Compose([
    tio.RandomAffine(),
    # tio.RescaleIntensity(out_min_max=(0, 1))
])
callbacks = [
    ModelCheckpoint(
        dirpath='./',
        monitor='val_loss',
        filename="model_DeepSurv",
        save_top_k=1,
        mode='min'),
    EarlyStopping(monitor='val_loss')
]

data_file    = np.load(sys.argv[1])
label_file   = sys.argv[2]
label_name   = sys.argv[3]

data,label   = LoadSortDataLabel(label_name, label_file, data_file)
trainer      = Trainer(gpus=1, max_epochs=20)
module_list  = nn.ModuleList(
    nn.Classifier2D(),
    nn.Classifier2D(),
    nn.Linear2D()
)

model        = MixModel(module_list)
dataloader   = DataModule(data, label, train_transform = train_transform, val_transform = val_transform, batch_size=4, inference=False)
trainer.fit(model, dataloader)

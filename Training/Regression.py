from pytorch_lightning import LightningDataModule, LightningModule
import numpy as np
import torch
from torch import nn
import torchvision
from torchvision import datasets, models, transforms
from torchvision import transforms
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import LightningDataModule, LightningModule, Trainer,seed_everything
from torchinfo import summary
import sys
import torchio as tio
import torchmetrics
import pandas as pd
## Module - Dataloaders
from DataGenerator.DataGenerator import DataModule, DataGenerator, PatientQuery
from DataGenerator.DataProcessing import LoadLabel
## Module - Models
from Models.Classifier2D import Classifier2D
from Models.Classifier3D import Classifier3D
from Models.Linear import Linear
from Models.MixModel import MixModel

## Main
train_transform = tio.Compose([
    tio.transforms.ZNormalization(),
    #tio.RandomAffine(),
    tio.RescaleIntensity(out_min_max=(0, 1))
])

val_transform = tio.Compose([
    tio.transforms.ZNormalization(),
    #tio.RandomAffine(),
    tio.RescaleIntensity(out_min_max=(0, 1))
])
callbacks = [
    ModelCheckpoint(
        dirpath='./',
        monitor='val_loss',
        filename="model_DeepSurv",
        save_top_k=1,
        mode='min'),
]

MasterSheet    = pd.read_csv(sys.argv[1],index_col='patid')
#analysis_inclusion=1,
#analysis_inclusion_rt=1) ## Query specific values in tags
label          = sys.argv[2]


clinical_columns = ["age","gender","race","ethnicity","zubrod","histology","nonsquam_squam","ajcc_stage_grp","pet_staging","rt_technique","has_egfr_hscore","egfr_hscore_200","smoke_hx","rx_terminated_ae","received_rt","rt_dose","overall_rt_review","fractionation_review","elapsed_days_review","tv_oar_review","gtv_review","ptv_review","ips_lung_review","contra_lung_review","spinal_cord_review","heart_review","esophagus_review","brachial_plexus_review","skin_review","dva_tv_review","dva_oar_review"]

RefColumns  = ["CTPath","DosePath"]
Label       = [label]

columns     = clinical_columns+RefColumns+Label
MasterSheet = MasterSheet[columns]
MasterSheet = MasterSheet.dropna(subset=["CTPath"])
MasterSheet = MasterSheet.dropna(subset=[label])
trainer     = Trainer(gpus=1, max_epochs=300)

## This is where you change how the data is organized
module_dict  = nn.ModuleDict({
    "Anatomy": Classifier3D(),
    "Dose": Classifier3D(),
    #"Clinical":Linear()
})

model        = MixModel(module_dict)
dataloader   = DataModule(MasterSheet, label, module_dict.keys(), train_transform = train_transform, val_transform = val_transform, batch_size=4, inference=False)
trainer.fit(model, dataloader)


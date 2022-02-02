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
import os
from DataGenerator.DataProcessing import LoadClincalData
from sklearn.preprocessing import StandardScaler, OneHotEncoder


## Main
train_transform = tio.Compose([
    tio.transforms.ZNormalization(),
    tio.RandomAffine(),
    tio.RandomFlip(),
    tio.RandomNoise(),
    tio.RandomMotion(),
    tio.transforms.Resize([40,40,40]),
    tio.RescaleIntensity(out_min_max=(0, 1))
])

val_transform = tio.Compose([
    tio.transforms.ZNormalization(),
    tio.transforms.Resize([40,40,40]),
    #tio.RandomAffine(),
    tio.RescaleIntensity(out_min_max=(0, 1))
])
callbacks = [
    ModelCheckpoint(
        dirpath='./',
        monitor='loss',
        filename="model_DeepSurv",
        save_top_k=1,
        mode='min'),
    EarlyStopping(monitor='val_loss',
                  check_finite=True),
]


MasterSheet    = pd.read_csv(sys.argv[1],index_col='patid')
#analysis_inclusion=1,
#analysis_inclusion_rt=1) ## Query specific values in tags
label          = sys.argv[2]

# For local test
#existPatient = os.listdir('C:/Users/clara/Documents/RTOG0617/nrrd_volumes')
#ids_common = set.intersection(set(MasterSheet.index.values), set(existPatient))
#MasterSheet = MasterSheet[MasterSheet.index.isin(ids_common)]



clinical_columns = ['arm', 'age', 'gender', 'race', 'ethnicity', 'zubrod',
                    'histology', 'nonsquam_squam', 'ajcc_stage_grp', 'rt_technique', # 'egfr_hscore_200', 'received_conc_cetuximab','rt_compliance_physician',
                    'smoke_hx', 'rx_terminated_ae', 'rt_dose',
                    'volume_ptv', 'dmax_ptv', 'v100_ptv',
                    'v95_ptv', 'v5_lung', 'v20_lung', 'dmean_lung', 'v5_heart',
                    'v30_heart', 'v20_esophagus', 'v60_esophagus', 'Dmin_PTV_CTV_MARGIN',
                    'Dmax_PTV_CTV_MARGIN', 'Dmean_PTV_CTV_MARGIN',
                    'rt_compliance_ptv90', 'received_conc_chemo',
                    ]
numerical_cols = ['age', 'volume_ptv', 'dmax_ptv', 'v100_ptv',
                      'v95_ptv', 'v5_lung', 'v20_lung', 'dmean_lung', 'v5_heart',
                      'v30_heart', 'v20_esophagus', 'v60_esophagus', 'Dmin_PTV_CTV_MARGIN',
                      'Dmax_PTV_CTV_MARGIN', 'Dmean_PTV_CTV_MARGIN']
category_cols = list(set(clinical_columns).difference(set(numerical_cols)))

#["age","gender","race","ethnicity","zubrod","histology","nonsquam_squam","ajcc_stage_grp","pet_staging","rt_technique","has_egfr_hscore","egfr_hscore_200","smoke_hx","rx_terminated_ae","received_rt","rt_dose","overall_rt_review","fractionation_review","elapsed_days_review","tv_oar_review","gtv_review","ptv_review","ips_lung_review","contra_lung_review","spinal_cord_review","heart_review","esophagus_review","brachial_plexus_review","skin_review","dva_tv_review","dva_oar_review"]

RefColumns  = ["CTPath","DosePath"]
Label       = [label]

columns     = clinical_columns+RefColumns+Label
MasterSheet = MasterSheet[columns]
MasterSheet = MasterSheet.dropna(subset=["CTPath"])
MasterSheet = MasterSheet.dropna(subset=category_cols)
MasterSheet = MasterSheet.dropna(subset=[label])
MasterSheet = MasterSheet.fillna(MasterSheet.mean())
trainer     = Trainer(gpus=1, max_epochs=20, callbacks=callbacks)

## This is where you change how the data is organized
module_dict  = nn.ModuleDict({
    #"Anatomy": Classifier3D(),
    "Dose": Classifier3D(),
    "Clinical": Linear()
})

numerical_data, category_data = LoadClincalData(MasterSheet)
sc = StandardScaler()
data1 = sc.fit_transform(numerical_data)

ohe = OneHotEncoder()
ohe.fit(category_data)
X_train_enc = ohe.transform(category_data)

model        = MixModel(module_dict)

dataloader   = DataModule(MasterSheet, label, module_dict.keys(), train_transform = train_transform, val_transform = val_transform, batch_size=4, numerical_norm = sc, category_norm = ohe, inference=False)
trainer.fit(model, dataloader)


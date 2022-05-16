import numpy as np
import torch
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
import sys, os
import torchio as tio
import pandas as pd

## Module - Dataloaders
from DataGenerator.DataGenerator import DataModule, DataGenerator, LoadClinicalData, QueryFromServer, SynchronizeData

## Module - Models
from Models.ModelCAE import ModelCAE
from Models.ModelTransUnet import ModelTransUnet
from Models.ModelCoTr import ModelCoTr
from Models.Classifier3D import Classifier3D
from Models.Linear import Linear
from Models.MixModel import MixModel

## Main
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import toml
from Utils.GenerateSmoothLabel import get_smoothed_label_distribution
from Utils.PredictionReports import PredictionReports

# from Utils.PredictionReport import generate_cumulative_dynamic_auc, classification_matrix, generate_report, plot_AUROC

config = toml.load(sys.argv[1])

logger = PredictionReports(config=config, save_dir='lightning_logs', name=config['MODEL']['BaseModel'])
logger.log_text()
img_dim = config['MODEL']['img_sizes']

train_transform = tio.Compose([
    tio.transforms.ZNormalization(),
    tio.RandomAffine(),
    tio.RandomFlip(),
    tio.RandomNoise(),
    tio.RandomMotion(),
    tio.transforms.Resize(img_dim),
    tio.RescaleIntensity(out_min_max=(0, 1))
])

val_transform = tio.Compose([
    tio.transforms.ZNormalization(),
    tio.transforms.Resize(img_dim),
    tio.RescaleIntensity(out_min_max=(0, 1))
])

filename = config['MODEL']['BaseModel'] + '_DeepSurv'


callbacks = [
    ModelCheckpoint(dirpath='./',
                    monitor='loss',
                    filename=filename,
                    save_top_k=1,
                    mode='min'),
    
    EarlyStopping(monitor='val_loss',
                  check_finite=True),
]

label = config['DATA']['target']

PatientList = QueryFromServer(config)
SynchronizeData(config, PatientList)
print(PatientList)

"""
clinical_columns = ['arm', 'age', 'gender', 'race', 'ethnicity', 'zubrod',
                    # 'histology', 'nonsquam_squam', 'ajcc_stage_grp', 'rt_technique',
                    'egfr_hscore_200', 'received_conc_cetuximab', 'rt_compliance_physician',
                    'smoke_hx', 'rx_terminated_ae', 'rt_dose',
                    'volume_ptv', 'rt_compliance_ptv90', 'received_conc_chemo',
                    ]
numerical_cols  = ['age', 'volume_ptv']
# , 'dmax_ptv', 'v100_ptv','v95_ptv', 'v5_lung', 'v20_lung', 'dmean_lung', 'v5_heart','v30_heart', 'v20_esophagus', 'v60_esophagus', 'Dmin_PTV_CTV_MARGIN','Dmax_PTV_CTV_MARGIN', 'Dmean_PTV_CTV_MARGIN'
category_cols = list(set(clinical_columns).difference(set(numerical_cols)))

# ["age","gender","race","ethnicity","zubrod","histology","nonsquam_squam","ajcc_stage_grp","pet_staging","rt_technique","has_egfr_hscore","egfr_hscore_200","smoke_hx","rx_terminated_ae","received_rt","rt_dose","overall_rt_review","fractionation_review","elapsed_days_review","tv_oar_review","gtv_review","ptv_review","ips_lung_review","contra_lung_review","spinal_cord_review","heart_review","esophagus_review","brachial_plexus_review","skin_review","dva_tv_review","dva_oar_review"]

RefColumns = ["CTPath", "DosePath"]
Label = [label]
"""
#columns = clinical_columns + RefColumns + Label

# trainer     =Trainer(accelerator="cpu", callbacks=callbacks)
## This is where you change how the data is organized
"""
numerical_data, category_data = LoadClinicalData(MasterSheet)
sc = StandardScaler()
data1 = sc.fit_transform(numerical_data)

ohe = OneHotEncoder()
ohe.fit(category_data)
X_train_enc = ohe.transform(category_data)
"""
module_dict = nn.ModuleDict()
module_selected = config['DATA']['module']

if config['MODEL']['BaseModel'] == 'CAE':
    Backbone = ModelCAE(config['MODEL'])

if config['MODEL']['BaseModel'] == 'TransUnet':
    Backbone = ModelTransUnet(config['MODEL'])

if config['MODEL']['BaseModel'] == 'CoTr':
    default_depth = 3
    img_sizes = []
    for i in range(default_depth):
        tmp = [x / 2 ** (i + 1) for x in img_dim]
        img_sizes.append(tmp)
    config['MODEL']['img_sizes'] = img_sizes
    Backbone = ModelCoTr(config['MODEL'])


if config['MODEL']['BaseModel'] == 'Unet':
    Backbone = Classifier3D()
if config['MODEL']['Clinical_Backbone']:
    Clinical_backbone = Linear()

for i, module in enumerate(module_selected):
    if module == 'Anatomy' or module == 'Dose':
        module_dict[module] = Backbone
    else:
        module_dict[module] = Clinical_backbone

dataloader = DataModule(PatientList, label, config, module_dict.keys(), train_transform=train_transform,
                        val_transform=val_transform, batch_size=config['MODEL']['batch_size'], numerical_norm=sc, category_norm=ohe,
                        inference=False)
train_label = dataloader.train_label

trainer = Trainer(gpus=1, max_epochs=3, logger=logger)  # callbacks=callbacks,
model = MixModel(module_dict, config, train_label, label_range=label_range, weights=weights)
trainer.fit(model, dataloader)

worstCase = 0
with torch.no_grad():
    outs = []
    for i, data in enumerate(dataloader.test_dataloader()):
        truth = data[1]
        x = data[0]
        output = model.test_step(data, i)
        outs.append(output)

    validation_labels = torch.cat([out['label'] for i, out in enumerate(outs)], dim=0)
    prediction_labels = torch.cat([out['prediction'] for i, out in enumerate(outs)], dim=0)
    prefix = 'test_'
    if config['MODEL']['Prediction_type'] == 'Regression':
        print('loss', model.loss_fcn(prediction_labels, validation_labels))
        if 'WorstCase' in config['REPORT']['matrix']:
            worst_record = logger.worst_case_show(outs, prefix)
            print('worst_AE', worst_record[prefix+'worst_AE'])
            if 'Anatomy' in config['DATA']['module']:
                text = 'test_worst_case_img'
                logger.log_image(worst_record[prefix+'worst_img'],text)
            if 'Dose' in config['DATA']['module']:
                text = 'test_worst_case_dose'
                logger.generate_report(worst_record[prefix+'worst_dose'],text)

    if config['MODEL']['Prediction_type'] == 'Classification':
        classification_out = logger.classification_matrix(prediction_labels.squeeze(), validation_labels, prefix)
        if 'ROC' in config['REPORT']['matrix']:
            logger.plot_AUROC(prediction_labels, validation_labels, prefix)
            print('AUROC:', classification_out[prefix + 'roc'])
        if 'Specificity' in config['REPORT']['matrix']:
            print('Specificity:', classification_out[prefix+'specificity'])

import numpy as np
import torch
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
import sys
import torchio as tio
import pandas as pd
## Module - Dataloaders
from DataGenerator.DataGenerator import DataModule, DataGenerator, PatientQuery
## Module - Models
from Models.ModelCAE import ModelCAE
from Models.ModelTransUnet import ModelTransUnet
from Models.ModelCoTr import ModelCoTr
import os
from DataGenerator.DataGenerator import LoadClincalData
from sklearn.preprocessing import StandardScaler, OneHotEncoder

torch.cuda.empty_cache()
## Main
from Models.Classifier3D import Classifier3D
from Models.Linear import Linear
import toml
from Models.MixModel import MixModel
from pytorch_lightning import loggers as pl_loggers
from Utils.GenerateSmoothLabel import get_smoothed_label_distribution
from Utils.PredictionReport import PredictionReport
# def main():

#config = toml.load('../SettingsCAE.ini')
config = toml.load(sys.argv[1])

tb_logger = pl_loggers.TensorBoardLogger(save_dir='lightning_logs', name=config['MODEL']['3D_MODEL'])
img_dim = config['DATA']['dim']

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
    # tio.RandomAffine(),
    tio.RescaleIntensity(out_min_max=(0, 1))
])
filename = config['MODEL']['3D_MODEL'] + '_DeepSurv'
callbacks = [
    ModelCheckpoint(
        dirpath='./',
        monitor='loss',
        filename=filename,
        save_top_k=1,
        mode='min'),
    EarlyStopping(monitor='val_loss',
                  check_finite=True),
]

path = config['DATA']['Path']
mastersheet = config['DATA']['Mastersheet']
target = config['DATA']['target']

MasterSheet = pd.read_csv(path + mastersheet, index_col='patid')
label = target

clinical_columns = ['arm', 'age', 'gender', 'race', 'ethnicity', 'zubrod',
                    #'histology', 'nonsquam_squam', 'ajcc_stage_grp', 'rt_technique',
                    'egfr_hscore_200', 'received_conc_cetuximab','rt_compliance_physician',
                    'smoke_hx', 'rx_terminated_ae', 'rt_dose',
                    'volume_ptv', 'rt_compliance_ptv90', 'received_conc_chemo',
                    ]
numerical_cols = ['age', 'volume_ptv']
#, 'dmax_ptv', 'v100_ptv','v95_ptv', 'v5_lung', 'v20_lung', 'dmean_lung', 'v5_heart','v30_heart', 'v20_esophagus', 'v60_esophagus', 'Dmin_PTV_CTV_MARGIN','Dmax_PTV_CTV_MARGIN', 'Dmean_PTV_CTV_MARGIN'
category_cols = list(set(clinical_columns).difference(set(numerical_cols)))

# ["age","gender","race","ethnicity","zubrod","histology","nonsquam_squam","ajcc_stage_grp","pet_staging","rt_technique","has_egfr_hscore","egfr_hscore_200","smoke_hx","rx_terminated_ae","received_rt","rt_dose","overall_rt_review","fractionation_review","elapsed_days_review","tv_oar_review","gtv_review","ptv_review","ips_lung_review","contra_lung_review","spinal_cord_review","heart_review","esophagus_review","brachial_plexus_review","skin_review","dva_tv_review","dva_oar_review"]

RefColumns = ["CTPath", "DosePath"]
Label = [label]

columns = clinical_columns + RefColumns + Label
MasterSheet = MasterSheet[columns]
MasterSheet = MasterSheet.dropna(subset=["CTPath"])
# MasterSheet = MasterSheet.dropna(subset=category_cols)
MasterSheet = MasterSheet.dropna(subset=[label])
MasterSheet = MasterSheet.fillna(MasterSheet.mean())

# MasterSheet[label] = (MasterSheet[label] > 24).astype(int)

if config['REGULARIZATION']['Label_smoothing']:
    weights, label_range = get_smoothed_label_distribution(MasterSheet, label)
else:
    weights = None
    label_range = None

# trainer     =Trainer(accelerator="cpu", callbacks=callbacks)
## This is where you change how the data is organized

numerical_data, category_data = LoadClincalData(MasterSheet)
sc = StandardScaler()
data1 = sc.fit_transform(numerical_data)

ohe = OneHotEncoder()
ohe.fit(category_data)
X_train_enc = ohe.transform(category_data)
patch_size = config['MODEL']['Patch_size']
embed_dim = config['MODEL']['Transformer_embed_size']  # For 2D image
batch_size = config['MODEL']['Batch_size']
num_layers = config['MODEL']['Transformer_layer']
dropout = config['MODEL']['Drop_rate']
mlp_dim = config['MODEL']['Transformer_mlp_dim']
module_selected = config['DATA']['module']
module_dict = nn.ModuleDict()
num_heads = config['MODEL']['Transformer_head']

if config['MODEL']['3D_MODEL'] == 'CAE':
    Backbone = ModelCAE(img_sizes=img_dim, patch_size=patch_size, embed_dim=embed_dim, in_channels=1,
                        num_layers=num_layers, num_heads=num_heads, dropout=dropout, mlp_dim=mlp_dim)

if config['MODEL']['3D_MODEL'] == 'TransUnet':
    embed_dim = patch_size ** 3 * 128
    img_size = [4, 16, 16]
    Backbone = ModelTransUnet(img_sizes=img_size, patch_size=patch_size, embed_dim=embed_dim, in_channels=128,
                              num_layers=num_layers, num_heads=num_heads, dropout=dropout, mlp_dim=mlp_dim)

if config['MODEL']['3D_MODEL'] == 'CoTr':
    default_depth = 3
    img_sizes = []
    for i in range(default_depth):
        tmp = [x / 2 ** (i + 1) for x in img_dim]
        img_sizes.append(tmp)
    Backbone = ModelCoTr(img_sizes=img_sizes, patch_size=patch_size, embed_dim=embed_dim, in_channels=1,
                         num_layers=num_layers, num_heads=num_heads, dropout=dropout, mlp_dim=mlp_dim)
if config['MODEL']['3D_MODEL'] == 'Unet':
    Backbone = Classifier3D()
if config['MODEL']['Clinical_Backbone']:
    Clinical_backbone = Linear()

for i, module in enumerate(module_selected):
    if module == 'Anatomy' or module == 'Dose':
        module_dict[module] = Backbone
    else:
        module_dict[module] = Clinical_backbone

dataloader = DataModule(MasterSheet, label, config, module_dict.keys(), train_transform=train_transform,
                        val_transform=val_transform, batch_size=batch_size, numerical_norm=sc, category_norm=ohe,
                        inference=False)
train_label = dataloader.train_label

trainer = Trainer(gpus=1, max_epochs=20, logger=tb_logger) #callbacks=callbacks,
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
    report = PredictionReport(prediction_labels, validation_labels)

    if config['MODEL']['Prediction_type'] == 'Regression':
        fig = report.generate_cumulative_dynamic_auc(train_label)
        model.logger.experiment.add_figure("AUC", fig)
        worst_record = report.worst_case_show(config, outs)
        print('worst_AE', worst_record['worst_AE'])
        if 'Anatomy' in config['DATA']['module']:
            grid_img = report.generate_report(worst_record['worst_img'])
            model.logger.experiment.add_image('validate_worst_case_img', grid_img)
        if 'Dose' in config['DATA']['module']:
            grid_dose = report.generate_report(worst_record['worst_dose'])
            model.logger.experiment.add_image('validate_worst_case_dose', grid_dose)

# with torch.no_grad():
#     output = trainer.test(model, dataloader.test_dataloader())
# print(output)


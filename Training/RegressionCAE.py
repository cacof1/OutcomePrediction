import torch
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import LightningDataModule, LightningModule, Trainer,seed_everything
import sys
import torchio as tio
import pandas as pd
## Module - Dataloaders
from DataGenerator.DataGenerator import DataModule, DataGenerator, PatientQuery
## Module - Models
from Models.ModelCAE import ModelCAE
import os
from DataGenerator.DataProcessing import LoadClincalData
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from Models.CNNEncoderForTransformer import CNNEncoderForTransformer
torch.cuda.empty_cache()
## Main
from Models.Classifier3D import Classifier3D
from Models.Linear import Linear
import toml
from Models.MixModel import MixModel
from Models.MixModelSmooth import MixModelSmooth
from pytorch_lightning import loggers as pl_loggers
from Utils.GenerateSmoothLabel import get_smoothed_label_distribution
from Utils.GenerateSmoothLabel import generate_report
tb_logger = pl_loggers.TensorBoardLogger(save_dir='lightning_logs', name='CAE')

config = toml.load('../Settings.ini')
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
    #tio.RandomAffine(),
    tio.RescaleIntensity(out_min_max=(0, 1))
])
callbacks = [
    ModelCheckpoint(
        dirpath='./',
        monitor='loss',
        filename="CAE_DeepSurv",
        save_top_k=1,
        mode='min'),
    EarlyStopping(monitor='val_loss',
                  check_finite=True),
]



path = config['DATA']['Path']
mastersheet = config['DATA']['Mastersheet']
target = config['DATA']['target']

MasterSheet    = pd.read_csv(path + mastersheet,index_col='patid')
#analysis_inclusion=1,
#analysis_inclusion_rt=1) ## Query specific values in tags
label          = target

# For local test
# existPatient = os.listdir('C:/Users/clara/Documents/RTOG0617/nrrd_volumes')
# ids_common = set.intersection(set(MasterSheet.index.values), set(existPatient))
# MasterSheet = MasterSheet[MasterSheet.index.isin(ids_common)]



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

if config['REGULARIZATION']['Label_smoothing']:
    weights, label_range = get_smoothed_label_distribution(MasterSheet, label)
else:
    weights = None
    label_range = None


trainer     = Trainer(gpus=1, max_epochs=20, callbacks=callbacks, logger=tb_logger) #
#trainer     =Trainer(accelerator="cpu", callbacks=callbacks)
## This is where you change how the data is organized

numerical_data, category_data = LoadClincalData(MasterSheet)
sc = StandardScaler()
data1 = sc.fit_transform(numerical_data)

ohe = OneHotEncoder()
ohe.fit(category_data)
X_train_enc = ohe.transform(category_data)
patch_size = config['MODEL']['Patch_size']
embed_dim = config['MODEL']['Transformer_embed_size']# For 2D image
batch_size = config['MODEL']['Batch_size']
num_layers = config['MODEL']['Transformer_layer']

module_selected = config['DATA']['module']
module_dict = nn.ModuleDict()

if config['MODEL']['3D_MODEL'] == 'CAE':
    Backbone = ModelCAE(config, img_sizes=img_dim, patch_size=patch_size, embed_dim=embed_dim, in_channels=1, num_layers=num_layers, weights=weights, label_range = label_range)
if config['MODEL']['Clinical_Backbone']:
    Clinical_backbone = Linear()

for i, module in enumerate(module_selected):
    if module == 'Anatomy' or module == 'Dose':
        module_dict[module] = Backbone
    else:
        module_dict[module] = Clinical_backbone
if config['REGULARIZATION']['smoothing']:
    model = MixModelSmooth(module_dict, config, label_range=label_range, weights=weights)
else:
    model = MixModel(module_dict, config)
dataloader = DataModule(MasterSheet, label, config, module_dict.keys(), train_transform=train_transform,
                        val_transform=val_transform, batch_size=batch_size, numerical_norm=sc, category_norm=ohe,
                        inference=False)

# trainer.fit(model, dataloader)

# worstCase = 0
#
# with torch.no_grad():
#     for i, data in enumerate(dataloader.test_dataloader()):
#         truth = data[1]
#         x = data[0]
#         output = model(x)
#         diff = torch.abs(output.flatten(0) - truth)
#         idx = torch.argmax(diff)
#         if diff[idx] > worstCase:
#             worst_img = x["Anatomy"][idx]
#         # output = model.test_step(data, i)
#         print('output:', output, 'true:', truth)
#
worstCase = 0
with torch.no_grad():
    for i, data in enumerate(dataloader.test_dataloader()):
        truth = data[1]
        x = data[0]
        if config['REGULARIZATION']['smoothing']:
            features = model(x, truth)
            output = model.classifier(features)
        else:
            output = model(x)
        diff = torch.abs(output.flatten(0) - truth)
        idx = torch.argmax(diff)
        if diff[idx] > worstCase:
            if 'Anatomy' in config['DATA']['module']:
                worst_img = data['Anatomy'][idx]
            if 'Dose' in config['DATA']['module']:
                worst_dose = data['Dose'][idx]
            worst_MAE = diff[idx]

        model.log('worst_MAE', worst_MAE)
        if 'Anatomy' in config['DATA']['module']:
            grid_img = generate_report(worst_img)
            model.logger.experiment.add_image('test_worst_case_img', grid_img, i)
        if 'Dose' in config['DATA']['module']:
            grid_dose = generate_report(worst_dose)
            model.logger.experiment.add_image('test_worst_case_dose', grid_dose, i)

# with torch.no_grad():
#     output = trainer.test(model, dataloader.test_dataloader())
#
# print(output)
import torch
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
import sys, os
import torchio as tio

torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)

## Module - Dataloaders
from DataGenerator.DataGenerator import DataModule, DataGenerator, LoadClinicalData, QueryFromServer, SynchronizeData

from Models.Classifier3D import Classifier3D
from Models.Classifier2D import Classifier2D
from Models.Linear import Linear
from Models.MixModel import MixModel

## Main
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import toml
from Utils.GenerateSmoothLabel import get_smoothed_label_distribution, get_module
from Utils.PredictionReports import PredictionReports
from pathlib import Path
from Utils.DicomTools import img_train_transform, img_val_transform

config = toml.load(sys.argv[1])
s_module = config['DATA']['module']
total_backbone = config['MODEL']['Prediction_type']
for module in s_module:
    total_backbone = total_backbone + '_' + module + '_' + config['MODEL'][module + '_Backbone']

train_transform = {}
val_transform = {}

for module in config['MODALITY'].keys():
    train_transform[module] = img_train_transform(config['DATA'][module + '_dim'])
    val_transform[module] = img_val_transform(config['DATA'][module + '_dim'])

ckpt_path = Path('./', total_backbone + '_ckpt')

callbacks = [
    ModelCheckpoint(dirpath=ckpt_path,
                    monitor='val_loss',
                    filename=total_backbone,
                    save_top_k=1,
                    mode='min'),
    # EarlyStopping(monitor='val_loss',
    #               check_finite=True),
]
label = config['DATA']['target']

module_dict = nn.ModuleDict()

PatientList = QueryFromServer(config)
SynchronizeData(config, PatientList)

if 'CT' in config['DATA']['module']:
    """
    if 'CT' in config['MODEL']['Finetune']:
        CT_config = toml.load(config['Finetune']['CT_config'])
        CT_module_dict = get_module(CT_config)
        CT_model = MixModel(CT_module_dict, CT_config)
        pretrained_CT_model = CT_model.load_from_checkpoint(checkpoint_path=config['Finetune']['CT_ckpt'], module_dict=CT_module_dict, config=CT_config)
    
        if config['MODEL']['CT_spatial_dims'] == 3:
            CT_Backbone = pretrained_CT_model.module_dict['CT'].model
        if config['MODEL']['CT_spatial_dims'] == 2:
            CT_Backbone = pretrained_CT_model.module_dict['CT'].backbone
        CT_Backbone.eval()
        for param in CT_Backbone.parameters():
            param.requires_grad = False
        module_dict['CT'] = CT_Backbone
    """
    # else:
    if config['MODEL']['CT_spatial_dims'] == 3:
        CT_Backbone = Classifier3D(config, 'CT')
    if config['MODEL']['CT_spatial_dims'] == 2:
        CT_Backbone = Classifier2D(config, 'CT')
    module_dict['CT'] = CT_Backbone

if 'Dose' in config['DATA']['module']:
    """
    if 'Dose' in config['MODEL']['Finetune']:
        Dose_config = toml.load(config['Finetune']['Dose_config'])
        Dose_module_dict = get_module(Dose_config)
        Dose_model = MixModel(Dose_module_dict, Dose_config)
        pretrained_Dose_model = Dose_model.load_from_checkpoint(checkpoint_path=config['Finetune']['Dose_ckpt'],
                                                                module_dict=Dose_module_dict, config=Dose_config)
        if config['MODEL']['Dose_spatial_dims'] == 3:
            Dose_Backbone = pretrained_Dose_model.module_dict['Dose'].model
        if config['MODEL']['Dose_spatial_dims'] == 2:
            Dose_Backbone = pretrained_Dose_model.module_dict['Dose'].backbone
        Dose_Backbone.eval()
        for param in Dose_Backbone.parameters():
            param.requires_grad = False
        module_dict['Dose'] = Dose_Backbone

    else:
    """
    if config['MODEL']['Dose_spatial_dims'] == 3:
        Dose_Backbone = Classifier3D(config, 'Dose')
    if config['MODEL']['Dose_spatial_dims'] == 2:
        Dose_Backbone = Classifier2D(config, 'Dose')
    module_dict['RTDose'] = Dose_Backbone

if config['MODEL']['Records_Backbone']:
    Clinical_backbone = Linear()

if 'Records' in config['DATA']['module']:
    module_dict['Records'] = Clinical_backbone
    PatientList, clinical_cols = LoadClinicalData(config, PatientList)
else:
    clinical_cols = None

if config['MODEL']['Prediction_type'] == 'Classification':
    threshold = config['MODEL']['Classification_threshold']
else:
    threshold = None

dataloader = DataModule(PatientList,
                        target=config['DATA']['target'],
                        selected_channel=module_dict.keys(),
                        dicom_folder=config['DATA']['DataFolder'],
                        train_transform=train_transform,
                        val_transform=val_transform,
                        batch_size=config['MODEL']['batch_size'],
                        threshold=threshold,
                        clinical_cols=clinical_cols
                        )

"""
if config['REGULARIZATION']['Label_smoothing']:
    weights, label_range = get_smoothed_label_distribution(PatientList, config)
else:
    weights = None
    label_range = None
"""

trainer = Trainer(gpus=torch.cuda.device_count(),
                  max_epochs=25,
                  logger=logger,
                  callbacks=callbacks)

model = MixModel(module_dict, config, label_range=None, weights=None)

roc_list =[]
for iter in range(1):
    filename = total_backbone
    logger = PredictionReports(config=config, save_dir='lightning_logs', name=filename)
    logger.log_text()
    trainer = Trainer(
                      # gpus=1,
                      accelerator="gpu",
                      devices=[0,1,2,3],
                      strategy=DDPStrategy(find_unused_parameters=True),
                      max_epochs=0,
                      logger=logger,
                      # callbacks=callbacks
                      )
    trainer.fit(model, dataloader)

    print('start testing...')
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
        roc_list.append(logger.report_test(config, outs, model, prediction_labels, validation_labels, prefix))
    print('finish test')

roc_avg = torch.mean(torch.tensor(roc_list))
print('avg_roc', str(roc_avg))

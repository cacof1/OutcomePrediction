import torch
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
# from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
import sys, os
import torchio as tio
import monai
from torchvision import transforms
torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)

## Module - Dataloaders
from DataGenerator.DataGenerator import *
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

# train_transform = {}
# val_transform = {}
#
# for module in config['MODALITY'].keys():
#     train_transform[module] = img_train_transform(config['DATA'][module + '_dim'])
#     val_transform[module] = img_val_transform(config['DATA'][module + '_dim'])

## 2D transform
train_transform = monai.transforms.Compose([    
    monai.transforms.ToTensor(),
    monai.transforms.AddChannel(),
    monai.transforms.NormalizeIntensity(),    
    monai.transforms.RandSpatialCrop(roi_size = [1,-1, -1], random_size = False),
    monai.transforms.SqueezeDim(dim=1),
    monai.transforms.ResizeWithPadOrCrop(spatial_size = config['DATA']['CT_dim']),
    monai.transforms.RepeatChannel(repeats=3),
    monai.transforms.RandAffine(),
    monai.transforms.RandHistogramShift(),
    monai.transforms.RandAdjustContrast(),
    monai.transforms.RandGaussianNoise(),

])

val_transform = monai.transforms.Compose([    
    monai.transforms.ToTensor(),
    monai.transforms.AddChannel(),
    monai.transforms.NormalizeIntensity(),    
    monai.transforms.RandSpatialCrop(roi_size = [1,-1,-1], random_size = False),
    monai.transforms.SqueezeDim(dim=1),
    monai.transforms.ResizeWithPadOrCrop(spatial_size = config['DATA']['CT_dim']),
    monai.transforms.RepeatChannel(repeats=3)

])

label = config['DATA']['target']

SubjectList = QuerySubjectList(config)
SynchronizeData(config, SubjectList)
SubjectInfo = QuerySubjectInfo(config, SubjectList)

c_norm = None
n_norm = None

dataloader = DataModule(SubjectList,
                        SubjectInfo,
                        config=config,
                        keys=module_dict.keys(),
                        train_transform = train_transform,
                        val_transform   = val_transform,
                        n_norm=n_norm,
                        c_norm=c_norm,
                        inference=False)

trainer = Trainer(gpus=torch.cuda.device_count(),
                  max_epochs=20,
                  logger=logger,
                  precision =16,
                  callbacks=callbacks)

model = MixModel(module_dict, config, label_range=None, weights=None)
logger.log_text()
trainer.fit(model, dataloader)


"""

module_dict = nn.ModuleDict()

if 'CT' in config['DATA']['module']:
    if config['MODEL']['CT_spatial_dims'] == 3:
        CT_Backbone = Classifier3D(config, 'CT')
    if config['MODEL']['CT_spatial_dims'] == 2:
        CT_Backbone = Classifier2D(config, 'CT')
    module_dict['CT'] = CT_Backbone

if 'Dose' in config['DATA']['module']:
    if config['MODEL']['Dose_spatial_dims'] == 3:
        Dose_Backbone = Classifier3D(config, 'Dose')
    if config['MODEL']['Dose_spatial_dims'] == 2:
        Dose_Backbone = Classifier2D(config, 'Dose')
    module_dict['RTDose'] = Dose_Backbone

if config['MODEL']['Records_Backbone']:
    Clinical_backbone = Linear()

if 'Records' in config['DATA']['module']:
    module_dict['Records'] = Clinical_backbone
    PatientList, clinical_cols = LoadClinicalData(config, SubjectList)
else:
    clinical_cols = None

if config['MODEL']['Prediction_type'] == 'Classification':
    threshold = config['DATA']['threshold']
else:
    threshold = None

ckpt_path = Path('./', total_backbone + '_ckpt')
roc_list = []
for iter in range(2):
    seed_everything(42)

    dataloader = DataModule(SubjectList,
                            SubjectInfo,
                            config=config,
                            keys=module_dict.keys(),
                            train_transform=train_transform,
                            val_transform=val_transform,
                            clinical_cols=clinical_cols,
                            inference=False)

    model = MixModel(module_dict, config, label_range=None, weights=None)
    model.apply(model.weights_reset)

    filename = total_backbone
    logger = PredictionReports(config=config, save_dir='lightning_logs', name=filename)
    logger.log_text()
    callbacks = [
        ModelCheckpoint(dirpath=ckpt_path,
                        monitor='val_loss',
                        filename='Iter_' + str(iter),
                        save_top_k=1,
                        mode='min'),
        # EarlyStopping(monitor='val_loss',
        #               check_finite=True),
    ]

    trainer = Trainer(
        gpus=1,
        # accelerator="gpu",
        # devices=[2,3],
        # strategy=DDPStrategy(find_unused_parameters=True),
        max_epochs=30,
        logger=logger,
        callbacks=callbacks
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
"""

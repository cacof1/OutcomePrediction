import torch
import torchvision
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
import sys, os
import torchio as tio
import monai
torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)

## Module - Dataloaders
from DataGenerator.DataGenerator import *
from Models.Classifier import Classifier
from Models.Linear import Linear
from Models.MixModel import MixModel

## Main
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import toml
from Utils.GenerateSmoothLabel import get_smoothed_label_distribution, get_module
from Utils.PredictionReports import PredictionReports
from pathlib import Path
from Utils.DicomTools import img_train_transform, img_val_transform
import torchio as tio
from torchmetrics import ConfusionMatrix
import torchmetrics

config = toml.load(sys.argv[1])
s_module = config['DATA']['module']

total_backbone = config['MODEL']['Prediction_type']
if config['DATA']['Multichannel']:
    module = 'Image'
    total_backbone = total_backbone + '_' + module + '_' + config['MODEL'][module + '_Backbone']
else:
    for module in s_module:
        total_backbone = total_backbone + '_' + module + '_' + config['MODEL'][module + '_Backbone']

# train_transform = {}
# val_transform = {}
#
# for module in config['MODALITY'].keys():
#     train_transform[module] = img_train_transform(config['DATA'][module + '_dim'])
#     val_transform[module] = img_val_transform(config['DATA'][module + '_dim'])

## 2D transform
train_transform = torchvision.transforms.Compose([
    monai.transforms.ScaleIntensity(),
    # monai.transforms.RandSpatialCrop(roi_size = [1,-1, -1], random_size = False),
    # monai.transforms.SqueezeDim(dim=1),
    monai.transforms.ResizeWithPadOrCrop(spatial_size = config['DATA']['CT_dim']),
    # monai.transforms.RepeatChannel(repeats=3),
    monai.transforms.RandAffine(),
    monai.transforms.RandHistogramShift(),
    monai.transforms.RandAdjustContrast(),
    monai.transforms.RandGaussianNoise(),

])

val_transform = torchvision.transforms.Compose([
    monai.transforms.ScaleIntensity(),
    # monai.transforms.RandSpatialCrop(roi_size = [1,-1,-1], random_size = False),
    # monai.transforms.SqueezeDim(dim=1),
    monai.transforms.ResizeWithPadOrCrop(spatial_size = config['DATA']['CT_dim']),
    # monai.transforms.RepeatChannel(repeats=3)
])

label = config['DATA']['target']

SubjectList = QuerySubjectList(config)
SynchronizeData(config, SubjectList)
SubjectInfo = QuerySubjectInfo(config, SubjectList)

module_dict = nn.ModuleDict()

if config['DATA']['Multichannel']:
    if config['MODALITY'].keys():
        CT_Backbone = Classifier(config, 'Image')
        module_dict['Image'] = CT_Backbone
else:
    if 'CT' in config['DATA']['module']:
        CT_Backbone = Classifier(config, 'CT')
        module_dict['CT'] = CT_Backbone

    if 'Dose' in config['DATA']['module']:
        Dose_Backbone = Classifier(config, 'Dose')
        module_dict['Dose'] = Dose_Backbone

    if 'PET' in config['DATA']['module']:
        Dose_Backbone = Classifier(config, 'PET')
        module_dict['PET'] = Dose_Backbone

if config['MODEL']['Records_Backbone']:
    Clinical_backbone = Linear()

if 'Records' in config['DATA']['module']:
    module_dict['Records'] = Clinical_backbone
    SubjectList, clinical_cols = LoadClinicalData(config, SubjectList)
else:
    clinical_cols = None

if config['MODEL']['Prediction_type'] == 'Classification':
    threshold = config['DATA']['threshold']
else:
    threshold = None

ckpt_path = Path('./', total_backbone + '_ckpt')
for iter in range(20):
    seed_everything(np.random.randint(1, 10000))
    # seed_everything(4200)
    dataloader = DataModule(SubjectList,
                            SubjectInfo,
                            config=config,
                            keys=config['DATA']['module'],
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
        # gpus=1,
        accelerator="gpu",
        devices=[2, 3],
        strategy=DDPStrategy(find_unused_parameters=False),
        max_epochs=30,
        logger=logger,
        callbacks=callbacks
    )
    trainer.fit(model, dataloader)
    torch.save({
        'state_dict': model.state_dict(),
    }, Path('ckpt_test', 'Iter_' + str(iter) + '.ckpt'))
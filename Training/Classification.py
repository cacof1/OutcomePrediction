import torch
import torchvision
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
import sys, os
import monai

torch.cuda.empty_cache()
## Module - Dataloaders
from DataGenerator.DataGenerator import *
from Models.Classifier import Classifier
from Models.Linear import Linear
from Models.MixModel import MixModel
from monai.transforms import EnsureChannelFirstd, ScaleIntensityd, ResampleToMatchd, BoundingRectd
## Main
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import toml
from Utils.GenerateSmoothLabel import get_smoothed_label_distribution, get_module
from Utils.PredictionReports import PredictionReports
from pathlib import Path  #
from torchmetrics import ConfusionMatrix
import torchmetrics

config = toml.load(sys.argv[1])


def threshold_at_one(x):
    return x > 0


## 2D transform
img_keys = list(config['MODALITY'].keys())

if config['MODALITY'].values():
    train_transform = torchvision.transforms.Compose([
        EnsureChannelFirstd(keys=img_keys),
        monai.transforms.CropForegroundd(keys=img_keys, source_key='Structs', select_fn=threshold_at_one),
        monai.transforms.Resized(keys=img_keys, spatial_size=config['DATA']['dim']),
        monai.transforms.ScaleIntensityd(keys=list(set(img_keys).difference(set(['Dose'])))),
        # monai.transforms.RandAffined(keys=img_keys),
        # monai.transforms.RandHistogramShiftd(keys=img_keys),
        # monai.transforms.RandAdjustContrastd(keys=img_keys),
        # monai.transforms.RandGaussianNoised(keys=img_keys),

    ])

    val_transform = torchvision.transforms.Compose([
        EnsureChannelFirstd(keys=img_keys),
        monai.transforms.CropForegroundd(keys=img_keys, source_key='Structs', select_fn=threshold_at_one),
        monai.transforms.Resized(keys=img_keys, spatial_size=config['DATA']['dim']),
        monai.transforms.ScaleIntensityd(list(set(img_keys).difference(set(['Dose'])))),
    ])
else:
    train_transform = None
    val_transform = None

## First Connect to XNAT
session = xnat.connect(config['SERVER']['Address'], user=config['SERVER']['User'],
                       password=config['SERVER']['Password'])
SubjectList = QuerySubjectList(config, session)
SynchronizeData(config, SubjectList)
SubjectList.dropna(subset=['xnat_subjectdata_field_map_survival_months'], inplace=True)

module_dict = nn.ModuleDict()
if config['DATA']['Multichannel']:  ## Single-Model Multichannel learning
    if config['MODALITY'].keys():
        module_dict['Image'] = Classifier(config, 'Image')
else:
    for key in config['MODALITY'].keys():  # Multi-Model Single Channel learning
        module_dict[key] = Classifier(config, key)

if 'Records' in config.keys():
    SubjectList, clinical_cols = LoadClinicalData(config, SubjectList)
    module_dict['Records'] = Linear(in_feat=len(clinical_cols), out_feat=42)
else:
    clinical_cols = None

## GeneratePath
for key in config['MODALITY'].keys():
    SubjectList[key + '_Path'] = ""
QuerySubjectInfo(config, SubjectList)
print(SubjectList)

for iter in range(0, 3, 1):
    seed_everything(np.random.randint(0, 10000), workers=True)
    dataloader = DataModule(SubjectList,
                            config=config,
                            keys=config['MODALITY'].keys(),
                            train_transform=train_transform,
                            val_transform=val_transform,
                            clinical_cols=clinical_cols,
                            inference=False)

    model = MixModel(module_dict, config)
    model.apply(model.weights_reset)
    filename = config['DATA']['LogFolder']

    logger = PredictionReports(config=config, save_dir='lightning_logs', name=filename)
    logger.log_text()
    logger._version = iter
    callbacks = [
        ModelCheckpoint(dirpath=Path(logger.log_dir, 'ckpt'),
                        monitor='val_loss',
                        filename='Iter_' + str(iter),
                        save_top_k=2,
                        mode='min'),
        # EarlyStopping(monitor='val_loss',
        #               check_finite=True),
    ]

    trainer = Trainer(
        accelerator="gpu",
        devices=[0, 1, 2, 3],
        strategy=DDPStrategy(find_unused_parameters=True),
        max_epochs=40,
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(model, dataloader)

with open(logger.root_dir + "/Config.ini", "w+") as toml_file:
    toml.dump(config, toml_file)
    toml_file.write("Train transform:\n")
    toml_file.write(str(train_transform))
    toml_file.write("Val/Test transform:\n")
    toml_file.write(str(val_transform))

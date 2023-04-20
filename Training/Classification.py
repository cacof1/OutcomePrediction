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
from monai.transforms import EnsureChannelFirstd, ScaleIntensityd, ResampleToMatchd
## Main
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import toml
from Utils.GenerateSmoothLabel import get_smoothed_label_distribution, get_module
from Utils.PredictionReports import PredictionReports
from pathlib import Path
from torchmetrics import ConfusionMatrix
import torchmetrics

config = toml.load(sys.argv[1])

## 2D transform
img_keys = list(config['MODALITY'].keys())
## Multichannel masks
#img_keys.remove('Structs')
#if 'Structs' in config['DATA'].keys():
#    for roi in config['DATA']['Structs']:
#         img_keys.append('Struct_' +  roi)

if config['MODALITY'].values():
    train_transform = torchvision.transforms.Compose([
        EnsureChannelFirstd(keys=img_keys),
        #ResampleToMatchd(list(set(img_keys).difference(set(['CT']))), key_dst='CT'),
        monai.transforms.ScaleIntensityd(keys=list(set(img_keys).difference(set(['Dose'])))),
        #monai.transforms.ResizeWithPadOrCropd(keys=img_keys, spatial_size=config['DATA']['dim']),
        monai.transforms.Resized(keys=img_keys, spatial_size=config['DATA']['dim']),
        monai.transforms.RandAffined(keys=img_keys),
        monai.transforms.RandHistogramShiftd(keys=img_keys),
        monai.transforms.RandAdjustContrastd(keys=img_keys),
        monai.transforms.RandGaussianNoised(keys=img_keys),

    ])

    val_transform = torchvision.transforms.Compose([
        EnsureChannelFirstd(keys=img_keys),
        #ResampleToMatchd(list(set(img_keys).difference(set(['CT']))), key_dst='CT'),
        monai.transforms.ScaleIntensityd(list(set(img_keys).difference(set(['Dose'])))),
        #monai.transforms.ResizeWithPadOrCropd(img_keys, spatial_size=config['DATA']['dim']),
        monai.transforms.Resized(keys=img_keys, spatial_size=config['DATA']['dim']),
    ])
else:
    train_transform = None
    val_transform = None

## First Connect to XNAT
session     = xnat.connect(config['SERVER']['Address'], user=config['SERVER']['User'],password=config['SERVER']['Password'])
SubjectList = QuerySubjectList(config, session)
SynchronizeData(config, SubjectList)
SubjectList.dropna(subset=['xnat_subjectdata_field_map_survival_months'], inplace=True)
# SubjectList.dropna(subset=['xnat_subjectdata_field_map_overall_stage'], inplace=True)

module_dict = nn.ModuleDict()
if config['DATA']['Multichannel']: ## Single-Model Multichannel learning
    if config['MODALITY'].keys():
        module_dict['Image'] = Classifier(config, 'Image')
else:
    for key in config['MODALITY'].keys():# Multi-Model Single Channel learning
        module_dict[key] = Classifier(config, key)

if 'Records' in config.keys():
    SubjectList, clinical_cols = LoadClinicalData(config, SubjectList)
    module_dict['Records'] = Linear(in_feat=len(clinical_cols), out_feat=42)
else:
    clinical_cols = None

## GeneratePath
for key in config['MODALITY'].keys():
    SubjectList[key+'_Path'] = ""
QuerySubjectInfo(config, SubjectList, session)
print(SubjectList)

# threshold = config['DATA']['threshold']
# ckpt_path = Path('./lightning_logs', total_backbone, 'ckpt')
rd = [53414, 88536, 89901, 62594, 13787, 21781, 18215, 4182, 10695, 61645, 93967, 35446, 41063, 98435, 94558, 67665,
      98831, 76684, 33670, 66239, 24417, 29551, 68018, 52785, 41160, 60264, 75053, 58354, 55180, 58358, 51182, 8260]

for iter in range(0, 15, 1):
    #seed_everything(rd[iter],workers=True)
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
    #full_ckpt_path = Path(ckpt_path, 'Iter_'+ str(iter) + '.ckpt')
    #model.load_state_dict(torch.load(full_ckpt_path)['state_dict'])

    filename = config['DATA']['LogFolder']

    logger = PredictionReports(config=config, save_dir='lightning_logs', name=filename)
    logger.log_text()
    logger._version = iter
    callbacks = [
        ModelCheckpoint(dirpath=Path(logger.log_dir, 'ckpt'),
                        monitor='val_loss_epoch',
                        filename='Iter_' + str(iter),
                        save_top_k=2,
                        mode='min'),
        # EarlyStopping(monitor='val_loss',
        #               check_finite=True),
    ]

    trainer = Trainer(
        #gpus=1,
        accelerator="gpu",
        devices=[0,1,2,3],
        strategy=DDPStrategy(find_unused_parameters=True),
        max_epochs=40,
        logger=logger,
        callbacks=callbacks,
    )
    #model = torch.compile(model)
    trainer.fit(model, dataloader)
    #torch.save({'state_dict': model.state_dict(),}, Path(logger.log_dir, 'Iter_' + str(iter) + '.ckpt'))
    
with open(logger.root_dir + "/Config.ini", "w+") as toml_file:
    toml.dump(config, toml_file)
    toml_file.write("Train transform:\n")
    toml_file.write(str(train_transform))
    toml_file.write("Val/Test transform:\n")
    toml_file.write(str(val_transform))


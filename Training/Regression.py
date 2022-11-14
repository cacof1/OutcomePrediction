import torch
import torchvision
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
# from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
import sys, os
import torchio as tio
import monai
torch.cuda.empty_cache()
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

total_backbone = ' '
## 2D transform
train_transform = torchvision.transforms.Compose([
    monai.transforms.ScaleIntensity(),
    # monai.transforms.RandSpatialCrop(roi_size = [1,-1, -1], random_size = False),
    # monai.transforms.SqueezeDim(dim=1),
    monai.transforms.ResizeWithPadOrCrop(spatial_size = config['DATA']['dim']),
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
    monai.transforms.ResizeWithPadOrCrop(spatial_size = config['DATA']['dim']),
    # monai.transforms.RepeatChannel(repeats=3)
])


## First Connect to XNAT
session = xnat.connect(config['SERVER']['Address'], user=config['SERVER']['User'],password=config['SERVER']['Password'])


SubjectList = QuerySubjectList(config, session)
## For testing
SubjectList = SubjectList.fillna(0)
SubjectList = SubjectList.sample(frac=1, random_state = 43)
SubjectList = SubjectList.head(30)
##
print(SubjectList)
SynchronizeData(config, SubjectList)
SubjectInfo = QuerySubjectInfo(config, SubjectList, session)

module_dict = nn.ModuleDict()
if config['DATA']['Multichannel']: ## Single-Model Multichannel learning
    if config['MODALITY'].keys():
        module_dict['Image'] = Classifier(config, 'Image')
else: 
    for key in config['MODALITY'].keys(): ## Multi-Model Single Channel learning
        module_dict[key] = Classifier(config, key)

if 'Records' in config['MODALITY'].keys():
    module_dict['Records'] = Linear()
    SubjectList, clinical_cols = LoadClinicalData(config, SubjectList)

else:
    clinical_cols = None

threshold = config['DATA']['threshold']
ckpt_path = Path('./', total_backbone + '_ckpt')
for iter in range(20):
    seed_everything(np.random.randint(1, 10000))
    dataloader = DataModule(SubjectList,
                            SubjectInfo,
                            config=config,
                            keys=config['MODALITY'].keys(),
                            train_transform=train_transform,
                            val_transform=val_transform,
                            clinical_cols=clinical_cols,
                            inference=False,
                            session = session)

    model = MixModel(module_dict, config)
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
        max_epochs=30,
        logger=logger,
        callbacks=callbacks
    )
    trainer.fit(model, dataloader)
    torch.save({'state_dict': model.state_dict(),}, Path('ckpt_test', 'Iter_' + str(iter) + '.ckpt'))

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
from Utils.DicomTools import img_train_transform, img_val_transform
import torchio as tio
from torchmetrics import ConfusionMatrix
import torchmetrics

config = toml.load(sys.argv[1])
s_module = config['DATA']['module']

total_backbone = config['MODEL']['Prediction_type']
if config['DATA']['Multichannel']:
   module = 'Image'
   total_backbone = total_backbone + '_' + module + '_' + config['MODEL']['Backbone']
else:
   for module in s_module:
       total_backbone = total_backbone + '_' + module + '_' + config['MODEL']['Backbone']
## 2D transform
#img_keys = list(config['MODALITY'].keys())
#if 'Mask' in config['DATA'].keys():
#    for roi in config['DATA']['Mask']:
#         img_keys.append('Mask_' +  roi)
img_keys = list(config['MODALITY'].keys())
if 'Mask' in config['DATA'].keys():
    img_keys.append('Mask')

train_transform = torchvision.transforms.Compose([
    EnsureChannelFirstd(keys=img_keys),
    ResampleToMatchd(list(set(config['MODALITY'].keys()).difference(set(['CT']))), key_dst='CT'),
    monai.transforms.ScaleIntensityd(keys=img_keys),
    # monai.transforms.ResizeWithPadOrCropd(keys=img_keys, spatial_size=config['DATA']['dim']),
    monai.transforms.Resized(keys=img_keys, spatial_size=config['DATA']['dim']),
    monai.transforms.RandAffined(keys=img_keys),
    monai.transforms.RandHistogramShiftd(keys=img_keys),
    monai.transforms.RandAdjustContrastd(keys=img_keys),
    monai.transforms.RandGaussianNoised(keys=img_keys),

])

val_transform = torchvision.transforms.Compose([
    EnsureChannelFirstd(keys=img_keys),
    ResampleToMatchd(list(set(config['MODALITY'].keys()).difference(set(['CT']))), key_dst='CT'),
    monai.transforms.ScaleIntensityd(img_keys),
    # monai.transforms.ResizeWithPadOrCropd(img_keys, spatial_size=config['DATA']['dim']),
    monai.transforms.Resized(keys=img_keys, spatial_size=config['DATA']['dim']),
])


## First Connect to XNAT
session = xnat.connect(config['SERVER']['Address'], user=config['SERVER']['User'],password=config['SERVER']['Password'])


SubjectList = QuerySubjectList(config, session)
## For testing
# SubjectList = SubjectList.fillna(0)
# SubjectList = SubjectList.sample(frac=1, random_state = 43)
# SubjectList = SubjectList.head(30)
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

if 'Records' in config['DATA']['module']:
    module_dict['Records'] = Linear()
    SubjectList, clinical_cols = LoadClinicalData(config, SubjectList)

else:
    clinical_cols = None

threshold = config['DATA']['threshold']
ckpt_path = Path('./', total_backbone + '_ckpt')
for iter in range(1,2,1):
    seed_everything(np.random.randint(1, 10000))
    dataloader = DataModule(SubjectList,
                            SubjectInfo,
                            config=config,
                            keys=config['DATA']['module'],
                            train_transform=train_transform,
                            val_transform=val_transform,
                            clinical_cols=clinical_cols,
                            inference=False,
                            session = session)

    model = MixModel(module_dict, config)
    model.apply(model.weights_reset)
    #full_ckpt_path = Path(ckpt_path, 'Iter_'+ str(iter) + '.ckpt')
    #model.load_state_dict(torch.load(full_ckpt_path)['state_dict'])

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
        #gpus=1,
        accelerator="gpu",
        devices=[0,1,2,3],
        strategy=DDPStrategy(find_unused_parameters=True),
        max_epochs=30,
        logger=logger,
        callbacks=callbacks
    )
    trainer.fit(model, dataloader)
    torch.save({'state_dict': model.state_dict(),}, Path('ckpt_test', 'Iter_' + str(iter) + '.ckpt'))

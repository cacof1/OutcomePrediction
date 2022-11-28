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
roc_list = []
sp_list = []
sensi_list = []
acc_list = []
pre_list = []

tprs = []
roc = torchmetrics.ROC()
auroc = torchmetrics.AUROC()
fig = plt.figure()
base_fpr = np.linspace(0, 1, 39)
cm = ConfusionMatrix(num_classes=2)
prediction_labels_full_list = []

for iter in range(0,1,1):
    # seed_everything(4200)
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
    # full_ckpt_path = Path(ckpt_path, 'Iter_'+ str(iter) + '.ckpt')
    # full_ckpt_path = Path('Classification_4_ckpt', 'Iter_' + str(iter) + '.ckpt')
    full_ckpt_path = 'ckpt_test/Iter_' + str(iter) + '.ckpt'
    model.load_state_dict(torch.load(full_ckpt_path)['state_dict'])
    # model.load_state_dict(torch.load(full_ckpt_path, map_location='cpu')['state_dict'])
    model.eval()
    print('start testing...')
    worstCase = 0
    with torch.no_grad():
        outs = []
        for i, data in enumerate(dataloader.test_dataloader()):
            truth = data[1]
            x = data[0]
            output = model.test_step(data, i)
            outs.append(output)

        validation_labels_full = torch.cat([out['label'] for i, out in enumerate(outs)], dim=0)
        prediction_labels_full = torch.cat([out['prediction'] for i, out in enumerate(outs)], dim=0)
        prediction_labels_full_list.append(prediction_labels_full.tolist())

prediction_labels = torch.tensor(prediction_labels_full_list).mean(dim=0)
validation_labels = validation_labels_full
roc = auroc(prediction_labels, validation_labels.int())
bcm = cm(prediction_labels.round(), validation_labels.int())
tn = bcm[0][0]
tp = bcm[1][1]
fp = bcm[0][1]
fn = bcm[1][0]
acc = bcm.diag().sum() / bcm.sum()
sensitivity = tp / (tp + fn)
precision = tp / (tp + fp)
spec = tn / (tn + fp)

print('avg_roc', str(roc))
print('avg_specificity', str(spec))
print('avg_sensitivity', str(sensitivity))
print('avg_accuracy', str(acc))
print('avg_precision', str(precision))
print('finish test')

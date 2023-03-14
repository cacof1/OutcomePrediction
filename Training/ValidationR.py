import torch
import torchvision
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
import sys, os
# import torchio as tio
import monai
from Utils.GenerateSmoothLabel import generate_cumulative_dynamic_auc
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
from Utils.PredictionReports import c_index, r2_index
# import torchio as tio
from torchmetrics import ConfusionMatrix
import torchmetrics
config = toml.load(sys.argv[1])
## 2D transform
img_keys = list(config['MODALITY'].keys())
# img_keys.remove('Structs')
# if 'Structs' in config['DATA'].keys():
#    for roi in config['DATA']['Structs']:
#         img_keys.append('Struct_' +  roi)

train_transform = torchvision.transforms.Compose([
    EnsureChannelFirstd(keys=img_keys),
    # ResampleToMatchd(list(set(img_keys).difference(set(['CT']))), key_dst='CT'),
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
    # ResampleToMatchd(list(set(img_keys).difference(set(['CT']))), key_dst='CT'),
    monai.transforms.ScaleIntensityd(img_keys),
    # monai.transforms.ResizeWithPadOrCropd(img_keys, spatial_size=config['DATA']['dim']),
    monai.transforms.Resized(keys=img_keys, spatial_size=config['DATA']['dim']),
])

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

print('clinical_cols:', len(clinical_cols))
## GeneratePath
for key in config['MODALITY'].keys():
    SubjectList[key + '_Path'] = ""
QuerySubjectInfo(config, SubjectList, session)

cindexs =[]
prediction_labels_full_list = []
# bidx = [21, 25, 0, 4, 6]
auroc = torchmetrics.AUROC(num_classes=1)
for iter in range(0, 20, 1):
    # iter = bidx[it]
    # seed_everything(4200)
    dataloader = DataModule(SubjectList,
                            config=config,
                            keys=config['MODALITY'].keys(),
                            train_transform=train_transform,
                            val_transform=val_transform,
                            clinical_cols=clinical_cols,
                            inference=False,
                            train_size=0.85)

    logger = PredictionReports(config=config, save_dir='lightning_logs/test', name= config['DATA']['LogFolder'])
    model = MixModel(module_dict, config)
    filename = 'lightning_logs/Regression/' + config['DATA']['LogFolder'] + '/version_' + str(iter)
    full_ckpt_path = Path(filename, 'Iter_' + str(iter) + '.ckpt')
    # full_ckpt_path = Path(ckpt_path, 'ckpt', 'Iter_' + str(iter) + '.ckpt')
    # full_ckpt_path = 'Iter_' + str(iter) + '.ckpt'
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
        cindex = c_index(prediction_labels_full, validation_labels_full.int())
        print('cindex_' + str(iter), cindex)
        cindexs.append(cindex[0])
        roc_i = auroc(torch.where(prediction_labels_full > 24, 1, 0), torch.where(validation_labels_full > 24, 1, 0))
        print('roc_' + str(iter), roc_i)
        prediction_labels_full_list.append(prediction_labels_full.tolist())
        logger.report_test(config, outs, model, prediction_labels_full, validation_labels_full, 'test_')

prediction_labels = torch.tensor(prediction_labels_full_list).mean(dim=0)
validation_labels = validation_labels_full

fig_path = 'lightning_logs/test/' + config['DATA']['LogFolder']
generate_cumulative_dynamic_auc(prediction_labels, validation_labels, fig_path)

cindex_tol = c_index(prediction_labels, validation_labels)
r2_tol = r2_index(prediction_labels, validation_labels)
print('cindex series', cindexs)
print('avg_roc', str(cindex_tol))
print('avg_r2', str(r2_tol))

auroc = auroc(torch.where(prediction_labels > 24, 1, 0), torch.where(validation_labels > 24, 1, 0))
print('auroc', auroc)

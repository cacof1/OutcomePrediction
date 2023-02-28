import torch
import torchvision
from torch import nn
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
import toml
from pathlib import Path
from torchmetrics import ConfusionMatrix
import torchmetrics
#import shap

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

## GeneratePath
for key in config['MODALITY'].keys():
    SubjectList[key + '_Path'] = ""
QuerySubjectInfo(config, SubjectList, session)

threshold = config['DATA']['threshold']
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
bidx = [21, 25, 0, 4, 6]

for it in range(0, 5, 1):
    iter = bidx[it]
    # seed_everything(4200)
    dataloader = DataModule(SubjectList,
                            config=config,
                            keys=config['MODALITY'].keys(),
                            train_transform=train_transform,
                            val_transform=val_transform,
                            clinical_cols=clinical_cols,
                            inference=False,
                            train_size=0.85)

    model = MixModel(module_dict, config)
    filename = 'lightning_logs/random_seed_75_Seg/version_' + str(iter)
    full_ckpt_path = Path(filename, 'Iter_' + str(iter) + '.ckpt')
    model.load_state_dict(torch.load(full_ckpt_path)['state_dict'])

    image = torch.cat([out for i, out in enumerate(dataloader.train_dataloader())], dim=0)
    label = torch.cat([out[1] for i, out in enumerate(dataloader.train_dataloader())], dim=0)

    to_explain = dataloader.test_dataloader()
    #e = shap.GradientExplainer((model, model.module_dict['Image'].backbone.features[7]), dataloader.train_dataloader())
    #shap_values, indexes = e.shap_values(to_explain, ranked_outputs=2, nsamples=200)

    # get the names for the classes
    # index_names = np.vectorize(lambda x: class_names[str(x)][1])(indexes)

    # plot the explanations
    #shap_values = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values]

    # shap.image_plot(shap_values, to_explain, index_names)



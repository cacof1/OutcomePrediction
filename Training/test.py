import torch
import torchvision
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
import sys, os
# import torchio as tio
import monai
import numpy as np
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
# import torchio as tio
from torchmetrics import ConfusionMatrix
import torchmetrics
import matplotlib.pyplot as plt
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
    #monai.transforms.ResizeWithPadOrCropd(keys=img_keys, spatial_size=config['DATA']['dim']),
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
    #monai.transforms.ResizeWithPadOrCropd(img_keys, spatial_size=config['DATA']['dim']),
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

# print('clinical_cols:', len(clinical_cols))
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
bidx = [1, 21, 25, 26, 30]
for iter in range(0, 12, 1):
    # iter = bidx[it]
    # seed_everything(4200)
    dataloader = DataModule(SubjectList,
                            config=config,
                            keys=config['MODALITY'].keys(),
                            train_transform=train_transform,
                            val_transform=val_transform,
                            clinical_cols=clinical_cols,
                            inference=False)

    logger = PredictionReports(config=config, save_dir='lightning_logs/test', name= config['DATA']['LogFolder'])
    model = MixModel(module_dict, config)
    filename = 'lightning_logs/' + config['DATA']['LogFolder'] + '/version_' + str(iter)
    # full_ckpt_path = Path(filename, 'Iter_' + str(iter) + '.ckpt')
    full_ckpt_path = Path(filename, 'ckpt', 'Iter_' + str(iter) + '.ckpt')
    model.load_state_dict(torch.load(full_ckpt_path)['state_dict'])

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

        validation_labels_full = torch.cat([out['label'][1] for i, out in enumerate(outs)], dim=0)
        validation_censor_full = torch.cat([out['label'][0] for i, out in enumerate(outs)], dim=0)
        prediction_labels_full = torch.cat([out['prediction'] for i, out in enumerate(outs)], dim=0)
        roc_i = auroc(prediction_labels_full, validation_labels_full.int())
        roc_list.append(roc_i)
        print('roc_' + str(iter), roc_i)
        fpr, tpr, _ = roc(prediction_labels_full, validation_labels_full)
        #if iter == 1:
        #    plt.plot(fpr, tpr, 'b', alpha=0.15, label='ROC of each bootstrap')
        #else:
        #    plt.plot(fpr, tpr, 'b', alpha=0.15)
        tpr = np.interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tprs.append(tpr)

        bcm = cm(prediction_labels_full.round(), validation_labels_full.int())
        tn = bcm[0][0]
        tp = bcm[1][1]
        fp = bcm[0][1]
        fn = bcm[1][0]

        acc = bcm.diag().sum() / bcm.sum()
        sensitivity = tp / (tp + fn)
        precision = tp / (tp + fp)
        spec = tn / (tn + fp)
        sp_list.append(spec)
        sensi_list.append(sensitivity)
        acc_list.append(acc)
        pre_list.append(precision)

        prediction_labels_full_list.append(prediction_labels_full.tolist())
        logger.report_test(config, outs, model, prediction_labels_full, [validation_censor_full, validation_labels_full], 'test_')
        with open(logger.log_dir + "/test_record.ini", "a") as toml_file:
            toml_file.write('\n')
            toml_file.write('label_iter_' + str(iter) + ':\n')
            toml_file.write(str(validation_labels_full))
            toml_file.write('\n')
            toml_file.write('censor_iter_' + str(iter) + ':\n')
            toml_file.write(str(validation_censor_full))
            toml_file.write('\n')
            toml_file.write('prediction_iter_' + str(iter) + ':\n')
            toml_file.write(str(prediction_labels_full))
            toml_file.write('\n')

prediction_labels = torch.tensor(prediction_labels_full_list).mean(dim=0)
validation_labels = validation_labels_full
mean_tprs = np.array(tprs).mean(axis=0)
std = np.array(tprs).std(axis=0)
tprs_upper = np.minimum(mean_tprs + std, 1)
tprs_lower = mean_tprs - std

plt.plot(base_fpr, mean_tprs, 'darkorange', label='Bootstrap-averaged ROC')
plt.legend(facecolor=[230/255, 236/255, 237/255])
plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)
plt.plot(base_fpr, base_fpr, 'r--')
plt.title('Test ROC')
plt.xlabel('False Positive Rate')
plt.xlim((0, 1))
plt.ylabel('True Positive rate')
plt.ylim((0, 1))
plt.show()

print('roc_list: ', roc_list)
print('spec:', sp_list)
print('sens:', sensi_list)
print('acc',acc_list)
fig.savefig('temp.png', dpi=fig.dpi,  transparent=True)

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

print('enm_roc', str(roc))
print('enm_specificity', str(spec))
print('enm_sensitivity', str(sensitivity))
print('enm_accuracy', str(acc))
print('enm_precision', str(precision))
print('finish test')


roc_avg = torch.mean(torch.tensor(roc_list))
sp_avg = torch.mean(torch.tensor(sp_list))
sensi_avg = torch.mean(torch.tensor(sensi_list))
acc_avg = torch.mean(torch.tensor(acc_list))
pre_avg = torch.mean(torch.tensor(pre_list))


roc_std = torch.std(torch.tensor(roc_list), unbiased=False)
sp_std = torch.std(torch.tensor(sp_list), unbiased=False)
sensi_std = torch.std(torch.tensor(sensi_list), unbiased=False)
acc_std = torch.std(torch.tensor(acc_list), unbiased=False)
pre_std = torch.std(torch.tensor(pre_list), unbiased=False)

print('avg_roc', str(roc_avg))
print('avg_specificity', str(sp_avg))
print('avg_sensitivity', str(sensi_avg))
print('avg_accuracy', str(acc_avg))
print('avg_precision', str(pre_avg))

print('std_roc', str(roc_std))
print('std_specificity', str(sp_std))
print('std_sensitivity', str(sensi_std))
print('std_accuracy', str(acc_std))
print('std_precision', str(pre_std))

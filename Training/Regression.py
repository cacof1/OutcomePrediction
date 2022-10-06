import torch
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
import sys, os
import torchio as tio
import monai
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

config = toml.load(sys.argv[1])

total_backbone = '' 
filename       = total_backbone
logger         = PredictionReports(config=config, save_dir='lightning_logs', name=filename)
logger.log_text()

img_dim = config['DATA']['dim']

## 2D transform
train_transform = monai.transforms.Compose([
    monai.transforms.NormalizeIntensity(),
    monai.transforms.RandSpatialCrop(roi_size = [1,-1, -1], random_size = False),
    monai.transforms.SqueezeDim(dim=1),
    monai.transforms.ResizeWithPadOrCrop(spatial_size = config['DATA']['dim']) 
])

val_transform = monai.transforms.Compose([
    monai.transforms.NormalizeIntensity(),
    monai.transforms.RandSpatialCrop(roi_size = [1,-1,-1], random_size = False),
    monai.transforms.SqueezeDim(dim=1), 
    monai.transforms.ResizeWithPadOrCrop(spatial_size = config['DATA']['dim']) 
])

ckpt_dirpath = Path('./', total_backbone + '_ckpt')
callbacks = [ModelCheckpoint(dirpath=ckpt_dirpath,
                             monitor='val_loss_' + str(getattr(torch.nn, config["MODEL"]["Loss_Function"])()),
                             filename=total_backbone,
                             save_top_k=1,
                             mode='min'),
]

module_dict = nn.ModuleDict()
module_dict['CT'] = Classifier2D(config)
SubjectList = QuerySubjectList(config)
SubjectList = SubjectList.head(20)
print(SubjectList)
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
                  callbacks=callbacks)

model = MixModel(module_dict, config, label_range=None, weights=None)
logger.log_text()
trainer.fit(model, dataloader)

"""
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
    if config['MODEL']['Prediction_type'] == 'Regression':
        logger.experiment.add_text('test loss: ', str(model.loss_fcn(prediction_labels, validation_labels)))
        logger.generate_cumulative_dynamic_auc(prediction_labels, validation_labels, 0, prefix)
        regression_out = logger.regression_matrix(prediction_labels, validation_labels, prefix)
        logger.experiment.add_text('test_cindex: ', str(regression_out[prefix + 'cindex']))
        logger.experiment.add_text('test_r2: ', str(regression_out[prefix + 'r2']))
        if 'WorstCase' in config['CHECKPOINT']['matrix']:
            worst_record = logger.worst_case_show(outs, prefix)
            logger.experiment.add_text('worst_test_AE: ', str(worst_record[prefix + 'worst_AE']))
            if 'CT' in config['DATA']['module']:
                text = 'test_worst_case_img'
                logger.log_image(worst_record[prefix + 'worst_img'], text)
            if 'Dose' in config['DATA']['module']:
                text = 'test_worst_case_dose'
                logger.log_image(worst_record[prefix + 'worst_dose'], text)

    if config['MODEL']['Prediction_type'] == 'Classification':
        classification_out = logger.classification_matrix(prediction_labels.squeeze(), validation_labels, prefix)
        if 'ROC' in config['CHECKPOINT']['matrix']:
            logger.plot_AUROC(prediction_labels, validation_labels, prefix)
            logger.experiment.add_text('test_AUROC: ', str(classification_out[prefix + 'roc']))
        if 'Specificity' in config['CHECKPOINT']['matrix']:
            logger.experiment.add_text('Specificity:', str(classification_out[prefix + 'specificity']))

print('finish test')
"""

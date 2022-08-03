import torch
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
import sys, os
import torchio as tio

## Module - Dataloaders
from DataGenerator.DataGenerator import DataModule, DataGenerator, LoadClinicalData, QueryFromServer, SynchronizeData

from Models.Classifier3D import Classifier3D
from Models.Classifier2D import Classifier2D
from Models.Linear import Linear
from Models.MixModel import MixModel

## Main
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import toml
from Utils.GenerateSmoothLabel import get_smoothed_label_distribution, get_module
from Utils.PredictionReports import PredictionReports

config = toml.load(sys.argv[1])
logger = PredictionReports(config=config, save_dir='lightning_logs', name=config['MODEL']['Backbone'])
logger.log_text()
img_dim = config['DATA']['dim']

train_transform = tio.Compose([
    tio.transforms.ZNormalization(),
    tio.RandomAffine(),
    tio.RandomFlip(),
    tio.RandomNoise(),
    tio.RandomMotion(),
    tio.transforms.Resize(img_dim),
    tio.RescaleIntensity(out_min_max=(0, 1))
])

val_transform = tio.Compose([
    tio.transforms.ZNormalization(),
    tio.transforms.Resize(img_dim),
    tio.RescaleIntensity(out_min_max=(0, 1))
])

filename = config['MODEL']['Backbone'] + '_DeepSurv'

callbacks = [
    ModelCheckpoint(dirpath='./',
                    monitor='train_loss',
                    filename=filename,
                    save_top_k=3,
                    mode='min'),

    EarlyStopping(monitor='val_loss',
                  check_finite=True),
]

label = config['DATA']['target']

module_dict = nn.ModuleDict()
s_module = config['DATA']['module']
if 'CT' in s_module:
    if 'CT' in config['MODEL']['Finetune']:
        CT_config = toml.load(config['Finetune']['CT_config'])
        CT_module_dict = get_module(CT_config)
        CT_model = MixModel(CT_module_dict, CT_config)
        pretrained_CT_model = CT_model.load_from_checkpoint(checkpoint_path=config['Finetune']['CT_ckpt'],
                                                        module_dict=CT_module_dict, config=CT_config)

        CT_Backbone = pretrained_CT_model.module_dict['CT'].model
        module_dict['CT'] = CT_Backbone
    else:
        CT_Backbone = Classifier3D(config)
        module_dict['CT'] = CT_Backbone

if 'Dose' in s_module:
    if 'Dose' in config['MODEL']['Finetune']:
        Dose_config = toml.load(config['Finetune']['Dose_config'])
        Dose_module_dict = get_module(Dose_config)
        Dose_model = MixModel(Dose_module_dict, Dose_config)
        pretrained_Dose_model = Dose_model.load_from_checkpoint(checkpoint_path=config['Finetune']['Dose_ckpt'],
                                                            module_dict=Dose_module_dict, config=Dose_config)

        Dose_Backbone = pretrained_Dose_model.module_dict['Dose'].model
        Dose_Backbone.eval()
        for param in Dose_Backbone.parameters():
            param.requires_grad = False
        module_dict['Dose'] = Dose_Backbone
    else:
        Dose_Backbone = Classifier3D(config)
        module_dict['Dose'] = Dose_Backbone

if config['MODEL']['Clinical_Backbone']:
    Clinical_backbone = Linear()

if 'Clinical' in s_module:
    module_dict['Clinical'] = Clinical_backbone


PatientList = QueryFromServer(config)
PatientList = [p for p in PatientList if p.label not in config['FILTER']['patient_id']]
SynchronizeData(config, PatientList)
print(PatientList)

if "Clinical" in config['DATA']['module']:
    category_feats, numerical_feats = LoadClinicalData(config, PatientList)
    n_norm = StandardScaler()
    n_norm.fit_transform(numerical_feats)
    c_norm = OneHotEncoder()
    c_norm.fit(category_feats)
else:
    c_norm = None
    n_norm = None
dataloader = DataModule(PatientList, config=config, keys=module_dict.keys(), train_transform=train_transform,
                        val_transform=val_transform, batch_size=config['MODEL']['batch_size'], numerical_norm=n_norm,
                        category_norm=c_norm,
                        inference=False)

if config['REGULARIZATION']['Label_smoothing']:
    weights, label_range = get_smoothed_label_distribution(PatientList, config)
else:
    weights = None
    label_range = None

ngpu = torch.cuda.device_count()
trainer = Trainer(gpus=ngpu, max_epochs=1, logger=logger, log_every_n_steps=10,
                  callbacks=callbacks)  # callbacks=callbacks,
model = MixModel(module_dict, config, label_range=label_range, weights=weights)

for param in model.parameters():
    print(param.requires_grad)

trainer.fit(model, dataloader)

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
        if 'WorstCase' in config['REPORT']['matrix']:
            worst_record = logger.worst_case_show(outs, prefix)
            logger.experiment.add_text('worst_test_AE: ', str(worst_record[prefix + 'worst_AE']))
            if 'Anatomy' in config['DATA']['module']:
                text = 'test_worst_case_img'
                logger.log_image(worst_record[prefix + 'worst_img'], text)
            if 'Dose' in config['DATA']['module']:
                text = 'test_worst_case_dose'
                logger.log_image(worst_record[prefix + 'worst_dose'], text)

    if config['MODEL']['Prediction_type'] == 'Classification':
        classification_out = logger.classification_matrix(prediction_labels.squeeze(), validation_labels, prefix)
        if 'ROC' in config['REPORT']['matrix']:
            logger.plot_AUROC(prediction_labels, validation_labels, prefix)
            logger.experiment.add_text('test_AUROC: ', str(classification_out[prefix + 'roc']))
        if 'Specificity' in config['REPORT']['matrix']:
            logger.experiment.add_text('Specificity:', str(classification_out[prefix + 'specificity']))

print('finish test')

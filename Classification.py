import torch
import torchvision
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
import sys, os
import monai
from pytorch_lightning.strategies import DDPStrategy
import numpy as np
import pandas as pd

## Module - Dataloaders
from DataGenerator.DataGenerator import DataModule
from Models.Classifier import Classifier
from Models.Linear import Linear
from Models.MixModel import MixModel
from monai.transforms import EnsureChannelFirstd, ScaleIntensityd, ResampleToMatchd, BoundingRectd
from Utils.DataExtraction import create_subject_list

## Main
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import toml
from Utils.PredictionReports import PredictionReports
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger


def threshold_at_one(x):
    return x > 2.1


def load_config():
    config = toml.load(sys.argv[1])
    return config


def transform_pipeline(config):
    img_keys = [k for k in config['MODALITY'].keys() if config['MODALITY'][k]]

    if config['MODALITY'].values():
        train_transform = torchvision.transforms.Compose([
            EnsureChannelFirstd(keys=img_keys + ['RTSTRUCT'] if 'RTSTRUCT' in config['MODALITY'].keys() else img_keys),
            monai.transforms.CropForegroundd(keys=img_keys, source_key='RTSTRUCT', select_fn=threshold_at_one),
            monai.transforms.Resized(keys=img_keys, spatial_size=config['DATA']['dim']),
            monai.transforms.RandAffined(keys=img_keys),
            monai.transforms.RandHistogramShiftd(keys=img_keys),
            monai.transforms.RandAdjustContrastd(keys=img_keys),
            monai.transforms.RandGaussianNoised(keys=img_keys),
            monai.transforms.ScaleIntensityd(keys=list(set(img_keys).difference(set(['RTDOSE'])))),
        ])

        val_transform = torchvision.transforms.Compose([
            EnsureChannelFirstd(keys=img_keys + ['RTSTRUCT'] if 'RTSTRUCT' in config['MODALITY'].keys() else img_keys),
            monai.transforms.CropForegroundd(keys=img_keys, source_key='RTSTRUCT', select_fn=threshold_at_one),
            monai.transforms.Resized(keys=img_keys, spatial_size=config['DATA']['dim']),
            monai.transforms.ScaleIntensityd(list(set(img_keys).difference(set(['RTDOSE'])))),
        ])
    else:
        train_transform = None
        val_transform = None

    return train_transform, val_transform


def build_model(config, clinical_cols):
    module_dict = nn.ModuleDict()
    if config['DATA']['multichannel']:  ## Single-Model Multichannel learning
        if config['MODALITY'].keys():
            module_dict['Image'] = Classifier(config, 'Image')
    else:
        for key in config['MODALITY'].keys():  # Multi-Model Single Channel learning
            if config['MODALITY'][key]:
                module_dict[key] = Classifier(config, key)
                if 'RTSTRUCT' in module_dict.keys():
                    module_dict.pop('RTSTRUCT')

    if 'Records' in config.keys():
        module_dict['Records'] = Linear(in_feat=len(clinical_cols), out_feat=42)

    return module_dict


def get_callbacks():
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename=f"{{model_name}}-epoch{{epoch:02d}}",
        save_top_k=1,
        mode='min')
    return [lr_monitor, checkpoint_callback]


def get_logger(config, model_name):
    logger_folder = config['DATA']['log_folder']
    logger = TensorBoardLogger(save_dir='lightning_logs', name=logger_folder)
    return logger


def main(config, rd):
    seed_everything(rd, workers=True)
    model_name = 'banana'
    SubjectList = create_subject_list(config)
    print(SubjectList)
    clinical_cols = config['DATA']['clinical_cols']
    logger = get_logger(config, model_name)
    callbacks = get_callbacks()
    train_transform, val_transform = transform_pipeline(config)
    module_dict = build_model(config, clinical_cols)
    model = MixModel(module_dict, config)
    # model.apply(model.weights_reset)

    dataloader = DataModule(SubjectList,
                            config=config,
                            keys=config['MODALITY'].keys(),
                            train_transform=train_transform,
                            val_transform=val_transform,
                            clinical_cols=clinical_cols,
                            rd=np.int16(rd),
                            inference=False)

    trainer = Trainer(
        accelerator="gpu",
        # devices=[0, 1, 2, 3],
        devices=[0],
        strategy=DDPStrategy(find_unused_parameters=True),
        max_epochs=40,
        logger=logger,
        log_every_n_steps=5,
        callbacks=callbacks,
    )

    trainer.fit(model, dataloader)

    with open(logger.log_dir + '/patient_list.ini', 'w+') as patient_list_file:
        patient_list_file.write('random_seed:\n')
        patient_list_file.write(str(rd))
        patient_list_file.write('\n')
        patient_list_file.write('train_patient_list:\n')
        patient_list_file.write('. '.join(list(dataloader.train_list['subject_label'])))
        patient_list_file.write('\n')
        patient_list_file.write('val_patient_list:\n')
        patient_list_file.write('. '.join(list(dataloader.val_list['subject_label'])))
        patient_list_file.write('\n')

    with open(logger.root_dir + "/Config.ini", "w+") as config_file:
        toml.dump(config, config_file)
        config_file.write("Train transform:\n")
        config_file.write(str(train_transform))
        config_file.write("Val/Test transform:\n")
        config_file.write(str(val_transform))


if __name__ == "__main__":
    config = (load_config()
              if len(sys.argv) > 1 else toml.load("/home/dgs1/Software/Miguel/OutcomePrediction/TestConfiguration.ini"))
    y = range(5)
    random_seed_list = []
    for i in y:
        rd = np.random.randint(10000)
        random_seed_list.append(rd)
        main(config, rd)

    print(random_seed_list)


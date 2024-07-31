import torch
import torchvision
from torch import nn
import torch.distributed as dist
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
import sys, os
import monai
from pytorch_lightning.strategies import DDPStrategy
import numpy as np
import pandas as pd
from pathlib import Path

## Module - Dataloaders
from DataGenerator.DataGenerator import DataModule
from Models.Classifier import Classifier
from Models.Linear import Linear
from Models.MixModel import MixModel
from monai.transforms import EnsureChannelFirstd, ScaleIntensityd, ResampleToMatchd, BoundingRectd
from Utils.DataExtraction import create_subject_list
from Utils.Transformations import StandardScalerd

## Main
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import toml
from Utils.PredictionReports import PredictionReports
from pathlib import Path
from pytorch_lightning.loggers import TensorBoardLogger


def is_rank_zero():
    return not dist.is_initialized() or dist.get_rank() == 0

def threshold_at_one(x):
    return x > 2.1


def load_config():
    config = toml.load(sys.argv[1])
    return config


def transform_pipeline(config):
    img_keys = [k for k in config['MODALITY'].keys() if config['MODALITY'][k]]
    records_keys = ['records'] if config['RECORDS']['records'] else []

    if len(records_keys) > 0 or len(img_keys) > 0:
        train_transform = []
        val_transform = []

        if len(records_keys) > 0:
            if 'continuous_cols' not in config['DATA'].keys():
                non_continuous = [config['DATA']['target'], config['DATA']['censor_label'],
                                  config['DATA']['subject_label']]
                config['DATA']['continuous_cols'] = [col for col in config['DATA']['clinical_cols']
                                                     if col not in non_continuous]
            train_transform += [
                StandardScalerd(keys=records_keys, continuous_variables=config['DATA']['continuous_cols']),]
            val_transform += [
                StandardScalerd(keys=records_keys, continuous_variables=config['DATA']['continuous_cols']),]

        if len(img_keys) > 0:
            condition = (('RTSTRUCT' not in config['MODALITY'].keys()) or (not config['MODALITY']['RTSTRUCT']) and
                         (config['MODALITY']['CT']) and ('CT' in config['MODALITY'].keys()))
            train_transform = [
                # EnsureChannelFirstd(keys=img_keys + ['RTSTRUCT'] if 'RTSTRUCT' in config['MODALITY'].keys() else img_keys),
                EnsureChannelFirstd(keys=img_keys + ['RTSTRUCT'] if condition else img_keys),
                monai.transforms.CropForegroundd(keys=img_keys, source_key='RTSTRUCT', select_fn=threshold_at_one),
                monai.transforms.Resized(keys=img_keys, spatial_size=config['DATA']['dim']),
                monai.transforms.RandAffined(keys=img_keys),
                monai.transforms.RandHistogramShiftd(keys=img_keys),
                monai.transforms.RandAdjustContrastd(keys=img_keys),
                monai.transforms.RandGaussianNoised(keys=img_keys),
                monai.transforms.ScaleIntensityd(keys=list(set(img_keys).difference(set(['RTDOSE'])))),
            ]

            val_transform = [
                # EnsureChannelFirstd(keys=img_keys + ['RTSTRUCT'] if 'RTSTRUCT' in config['MODALITY'].keys() else img_keys),
                EnsureChannelFirstd(keys=img_keys + ['RTSTRUCT'] if condition else img_keys),
                monai.transforms.CropForegroundd(keys=img_keys, source_key='RTSTRUCT', select_fn=threshold_at_one),
                monai.transforms.Resized(keys=img_keys, spatial_size=config['DATA']['dim']),
                monai.transforms.ScaleIntensityd(list(set(img_keys).difference(set(['RTDOSE'])))),
            ]

        train_transform = torchvision.transforms.Compose(train_transform)
        val_transform = torchvision.transforms.Compose(val_transform)
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

    if 'RECORDS' in config.keys() and config['RECORDS']['records']:
        module_dict['records'] = Linear(config, in_feat=len(clinical_cols),
                                        out_feat=config['MODEL']['linear_out'])

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
    SubjectList.to_csv(Path(config['DATA']['log_folder'])/'data_table.csv')
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
        devices=config['MODEL']['devices'],
        strategy=DDPStrategy(find_unused_parameters=True),
        max_epochs=config['MODEL']['max_epochs'],
        logger=logger,
        log_every_n_steps=1,
        callbacks=callbacks,
    )

    # Ensure the directory exists only on rank 0
    if is_rank_zero():
        if not Path(logger.log_dir).exists():
            Path(logger.log_dir).mkdir(parents=True)
        with open(logger.log_dir + '/patient_list.ini', 'w+') as patient_list_file:
            patient_list_file.write('random_seed:\n')
            patient_list_file.write(str(rd))
            patient_list_file.write('\n')
            patient_list_file.write('train_patient_list:\n')
            patient_list_file.write('. '.join(list(dataloader.train_list[config['DATA']['subject_label']])))
            patient_list_file.write('\n')
            patient_list_file.write('val_patient_list:\n')
            patient_list_file.write('. '.join(list(dataloader.val_list[config['DATA']['subject_label']])))
            patient_list_file.write('\n')

        with open(logger.root_dir + "/Config.ini", "w+") as config_file:
            toml.dump(config, config_file)
            config_file.write("Train transform:\n")
            config_file.write(str(train_transform))
            config_file.write("Val/Test transform:\n")
            config_file.write(str(val_transform))

    trainer.fit(model, dataloader)


if __name__ == "__main__":
    config = (load_config()
              if len(sys.argv) > 1 else toml.load("/OPConfiguration.ini"))
    y = range(config['RUN']['bootstrap_n'])
    if 'random_state' in config['RUN'].keys():
        np.random.seed(seed=config['RUN']['random_state'])
    random_seed_list = np.random.randint(10000, size=len(y))
    print(random_seed_list)
    for i in y:
        main(config, random_seed_list[i])

    print(random_seed_list)


from abc import ABC

import matplotlib.pyplot as plt
import numpy as np
from torchinfo import summary
import torchmetrics
from monai.networks import blocks, nets
import torch
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
import sys, os
import torchio as tio
torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)
## Module - Dataloaders
from DataGenerator.DataGenerator import DataModule, DataGenerator, QueryFromServer, SynchronizeData
from Models.MixModel import MixModel

## Main
import toml
from Utils.PredictionReports import PredictionReports
from pathlib import Path


## Model
class AutoEncoder3D(LightningModule, ABC):
    def __init__(self, config):
        super().__init__()
        self.n_classes = 1
        module_str = config['MODEL']['Backbone']
        parameters = config['MODEL_PARAMETERS']
        self.config = config

        model_str = 'nets.' + module_str + '(**parameters)'
        self.model = eval(model_str)
        # only use network for features

        summary(self.model.to('cuda'), (config['MODEL']['batch_size'], 1, *config['DATA']['dim']))
        self.accuracy = torchmetrics.AUC(reorder=True)
        self.loss_fcn = torch.nn.MSELoss()

    def forward(self, datadict):
        key = list(datadict.keys())[0]
        return self.model(datadict[key])

    def training_step(self, batch, batch_idx):
        image_dict, label = batch
        prediction = self.forward(image_dict)
        key = list(image_dict.keys())[0]
        loss = self.loss_fcn(prediction, image_dict[key])
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        image_dict, label = batch
        prediction = self.forward(image_dict)
        key = list(image_dict.keys())[0]
        loss = self.loss_fcn(prediction, image_dict[key])

        if batch_idx < 3:
            im_st = image_dict[key][0, 0, 5, :, :]
            plt.imshow(im_st.cpu())
            plt.title('standard image')
            plt.show()
            im = prediction[0, 0, 5, :, :]
            plt.imshow(im.cpu())
            plt.title('generated image')
            plt.show()
        self.log("val_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        image_dict, label = batch
        prediction = self.forward(image_dict)
        im = prediction[0, 0, 5, :, :].cpu()
        plt.imshow(im)
        plt.show()
        return prediction

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return [optimizer], [scheduler]


if __name__ == "__main__":
    config = toml.load(sys.argv[1])
    s_module = config['DATA']['module']
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

    label = config['DATA']['target']

    module_dict = nn.ModuleDict()

    s_module = config['DATA']['module'][0]
    module_dict[s_module] = AutoEncoder3D(config)

    PatientList = QueryFromServer(config)
    PatientList = [p for p in PatientList if p.label not in config['FILTER']['patient_id']]
    SynchronizeData(config, PatientList)
    print(PatientList)

    dataloader = DataModule(PatientList, config=config, keys=module_dict.keys(), train_transform=train_transform,
                            val_transform=val_transform, batch_size=config['MODEL']['batch_size'],
                            numerical_norm=None,
                            category_norm=None,
                            inference=False)

    for iter in range(1):
        total_backbone = config['MODEL']['Prediction_type'] + '_' + s_module + '_' + config['MODEL']['Backbone']
        logger = PredictionReports(config=config, save_dir='lightning_logs', name=total_backbone)
        logger.log_text()

        # ckpt_dirpath = Path('./', total_backbone + '_ckpt')
        #
        # callbacks = [
        #     ModelCheckpoint(dirpath=ckpt_dirpath,
        #                     monitor='val_loss',
        #                     filename='Iter',
        #                     save_top_k=1,
        #                     mode='min'),
        # ]

        ngpu = torch.cuda.device_count()
        trainer = Trainer(gpus=1, max_epochs=15, logger=logger, log_every_n_steps=10, #callbacks=callbacks,
                          auto_lr_find=True)
        model = AutoEncoder3D(config)

        i = 0
        for param in model.parameters():
            i = i + 1
            print(param.requires_grad)

        trainer.fit(model, dataloader)

        print('start testing...')
        worstCase = 0
        with torch.no_grad():
            outs = []
            for i, data in enumerate(dataloader.test_dataloader()):
                truth = data[1]
                x = data[0]
                output = model.predict_step(data, i)
                key = list(x.keys())[0]
                im_st = x[key][0, 0, 5, :, :]
                plt.imshow(im_st.cpu())
                plt.title('standard image')
                plt.show()
        print('finish test')


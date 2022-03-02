import matplotlib.pyplot as plt
from pytorch_lightning import LightningDataModule, LightningModule
import numpy as np
import torch
from torch import nn
from collections import Counter
import torchvision
from torchvision import datasets, models, transforms
from torchvision import transforms
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
import sys
import torchio as tio
import sklearn
from pytorch_lightning import loggers as pl_loggers
import torchmetrics
from Utils.GenerateSmoothLabel import generate_report
from Models.fds import FDS
## Models
from Models.Linear import Linear
from Models.Classifier2D import Classifier2D
from Models.Classifier3D import Classifier3D


class MixModelSmooth(LightningModule):
    def __init__(self, module_dict, config, label_range, weights, loss_fcn=torch.nn.BCEWithLogitsLoss()):
        super().__init__()
        self.module_dict = module_dict
        self.config = config
        self.label_range = label_range
        self.weights = weights
        self.FDS = FDS(feature_dim=1032, start_update=0, start_smooth=1, kernel='gaussian', ks=7, sigma=3)
        self.classifier = nn.Sequential(
            nn.LazyLinear(128),
            nn.LazyLinear(1)
        )
        self.accuracy = torchmetrics.AUC(reorder=True)
        self.loss_fcn = torch.nn.MSELoss()  # loss_fcn

    def WeightedMSE(self, prediction, labels):
        loss = 0
        for i, label in enumerate(labels):
            idx = (self.label_range == int(label.cpu().numpy())).nonzero()
            if (idx is not None) and (idx[0][0] < 60):
                # print(idx[0][0])
                loss = loss + (prediction[i] - label) ** 2 * self.weights[idx[0][0]]
            else:
                loss = loss + (prediction[i] - label) ** 2 * self.weights[-1]
        loss = loss / (i + 1)
        return loss

    def forward(self, datadict, labels):
        features = torch.cat([self.module_dict[key](datadict[key]) for key in self.module_dict.keys()], dim=1)
        if self.config['REGULARIZATION']['Feature_smoothing']:
            if self.training and self.current_epoch >= 1:
                features = self.FDS.smooth(features, labels, self.current_epoch)

        return features

    def training_step(self, batch, batch_idx):
        datadict, label = batch
        features = self.forward(datadict, label)
        prediction = self.classifier(features)
        print(prediction, label)
        if self.config['REGULARIZATION']['Label_smoothing']:
            loss = self.WeightedMSE(prediction.squeeze(dim=1), batch[-1])
        else:
            loss = self.loss_fcn(prediction.squeeze(dim=1), batch[-1])
        self.log("loss", loss, on_epoch=True)
        out = {'loss': loss, 'features': features, 'label': label}
        return out

    def training_epoch_end(self, training_step_outputs):
        if self.config['REGULARIZATION']['Feature_smoothing']:
            training_features = torch.cat([out['features'] for i, out in enumerate(training_step_outputs)], dim=0)
            training_labels = torch.cat([out['label'] for i, out in enumerate(training_step_outputs)], dim=0)
            if self.current_epoch >= 0:
                self.FDS.update_last_epoch_stats(self.current_epoch)
                self.FDS.update_running_stats(training_features, training_labels, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        out = {}
        datadict, label = batch
        prediction = self.classifier(self.forward(datadict, label))
        val_loss = self.loss_fcn(prediction.squeeze(dim=1), batch[-1])
        self.log("val_loss", val_loss, on_epoch=True)
        MAE = torch.abs(prediction.flatten(0) - label)
        out['MAE'] = MAE
        if 'Dose' in self.config['DATA']['module']:
            out['dose'] = datadict['Dose']
        if 'Anatomy' in self.config['DATA']['module']:
            out['img'] = datadict['Anatomy']
        return out

    def validation_epoch_end(self, validation_step_outputs):
        worst_MAE = 0
        for i, data in enumerate(validation_step_outputs):
            loss = data['MAE']
            idx = torch.argmax(loss)
            if loss[idx] > worst_MAE:
                if 'Anatomy' in self.config['DATA']['module']:
                    worst_img = data['img'][idx]
                if 'Dose' in self.config['DATA']['module']:
                    worst_dose = data['dose'][idx]
                worst_MAE = loss[idx]
        self.log('worst_MAE', worst_MAE)
        if 'Anatomy' in self.config['DATA']['module']:
            grid_img = generate_report(worst_img)
            self.logger.experiment.add_image('validate_worst_case_img', grid_img, self.current_epoch)
        if 'Dose' in self.config['DATA']['module']:
            grid_dose = generate_report(worst_dose)
            self.logger.experiment.add_image('validate_worst_case_dose', grid_dose, self.current_epoch)

    def test_step(self, batch, batch_idx):
        datadict, label = batch
        prediction = self.classifier(self.forward(datadict, label))
        test_loss = self.loss_fcn(prediction.squeeze(dim=1), batch[-1])
        self.log('test_loss', test_loss)
        return test_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return [optimizer], [scheduler]


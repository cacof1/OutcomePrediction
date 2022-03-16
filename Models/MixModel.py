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

## Models
from Models.Linear import Linear
from Models.Classifier2D import Classifier2D
from Models.Classifier3D import Classifier3D
from Utils.GenerateSmoothLabel import generate_cumulative_dynamic_auc
from sksurv.metrics import concordance_index_censored
class MixModel(LightningModule):
    def __init__(self, module_dict, config, loss_fcn=torch.nn.BCEWithLogitsLoss()):
        super().__init__()
        self.module_dict = module_dict
        self.config = config
        self.classifier = nn.Sequential(
            nn.LazyLinear(128),
            nn.LazyLinear(1)
        )
        self.accuracy = torchmetrics.AUC(reorder=True)
        self.loss_fcn = torch.nn.MSELoss()  # loss_fcn

    def forward(self, datadict):
        # module_dict = nn.ModuleDict()
        features = torch.cat([self.module_dict[key](datadict[key]) for key in self.module_dict.keys()], dim=1)
        # for key in self.module_dict.keys():
        #     data = self.module_dict[key](datadict[key])
        #     print(data.shape)

        return self.classifier(features)

    def training_step(self, batch, batch_idx):
        self.train_label = []
        datadict, label = batch
        prediction = self.forward(datadict)
        print(prediction, label)
        self.train_label.extend([float(i) for i in label])
        loss = self.loss_fcn(prediction.squeeze(dim=1), batch[-1])

        SSres = loss * batch[-1].shape[0]
        SStotal = torch.sum(torch.square(batch[-1] - torch.mean(batch[-1])))
        r2 = 1 - SSres / SStotal

        event_indicator = torch.ones(batch[-1].shape, dtype=torch.bool)
        risk = 1 / prediction.squeeze(dim=1)
        cindex = concordance_index_censored(event_indicator.cpu().detach().numpy(), event_time=batch[-1].cpu().detach().numpy(), estimate=risk.cpu().detach().numpy())
        self.log('cindex', cindex[0], on_epoch=True)
        self.log('r2', r2, on_epoch=True)
        self.log("loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        out = {}
        datadict, label = batch
        prediction = self.forward(datadict)
        val_loss = self.loss_fcn(prediction.squeeze(dim=1), batch[-1])
        SSres = val_loss * batch[-1].shape[0]
        SStotal = torch.sum(torch.square(batch[-1] - torch.mean(batch[-1])))
        r2 = 1 - SSres / SStotal

        event_indicator = torch.ones(batch[-1].shape, dtype=torch.bool)
        risk = 1 / prediction.squeeze(dim=1)
        cindex = concordance_index_censored(event_indicator.cpu(), event_time=batch[-1].cpu(), estimate=risk.cpu())
        self.log('cindex', cindex[0], on_epoch=True)

        self.log("val_loss", val_loss, on_epoch=True)
        self.log('r2', r2, on_epoch=True)

        MAE = torch.abs(prediction.flatten(0) - label)
        out['MAE'] = MAE
        out['prediction'] = prediction.squeeze(dim=1)
        out['label'] = label 
        if 'Dose' in self.config['DATA']['module']:
            out['dose'] = datadict['Dose']
        if 'Anatomy' in self.config['DATA']['module']:
            out['img'] = datadict['Anatomy']
        return out

    def validation_epoch_end(self, validation_step_outputs):

        validation_labels = torch.cat([out['label'] for i, out in enumerate(validation_step_outputs)], dim=0)
        risk_score = 1 / torch.cat([out['prediction'] for i, out in enumerate(validation_step_outputs)], dim=0)
        fig = generate_cumulative_dynamic_auc(torch.FloatTensor(self.train_label), validation_labels, risk_score)
        self.logger.experiment.add_figure("AUC", fig, self.current_epoch)

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
        prediction = self.forward(datadict)
        test_loss = self.loss_fcn(prediction.squeeze(dim=1), batch[-1])
        # print('test_prediction:', prediction, label)
        self.log('test_loss', test_loss)
        return test_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return [optimizer], [scheduler]

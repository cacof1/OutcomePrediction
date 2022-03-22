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
from pytorch_lightning import loggers as pl_loggers
import torchmetrics
# from Utils.GenerateSmoothLabel import generate_report, plot_AUROC
from Models.fds import FDS
from Losses.loss import WeightedMSE, CrossEntropy
from sksurv.metrics import concordance_index_censored
from Utils.PredictionReport import PredictionReport


class MixModel(LightningModule):
    def __init__(self, module_dict, config, train_label, label_range, weights, loss_fcn=torch.nn.BCEWithLogitsLoss()):
        super().__init__()
        self.FDS = None
        self.module_dict = module_dict
        self.config = config
        self.label_range = label_range
        self.weights = weights
        # self.FDS = FDS(feature_dim=1032, start_update=0, start_smooth=1, kernel='gaussian', ks=7, sigma=3)
        if config['MODEL']['Prediction_type'] == 'Classification':
            self.classifier = nn.Sequential(
                nn.LazyLinear(128),
                nn.LazyLinear(1),
                nn.Sigmoid()
            )
            self.loss_fcn = torch.nn.BCELoss()
            self.roc = torchmetrics.ROC()
            self.auc = torchmetrics.AUC(reorder=True)
            self.accuracy = torchmetrics.AUROC()
            self.specificity = torchmetrics.Specificity()

        if config['MODEL']['Prediction_type'] == 'Regression':
            self.classifier = nn.Sequential(
                nn.LazyLinear(128),
                nn.LazyLinear(1)
            )
            self.loss_fcn = torch.nn.MSELoss()

        self.train_label = train_label
        self.report = PredictionReport()

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
        # self.train_label = []
        #self.train_label.extend([float(i) for i in label])
        loss = self.loss_fcn(prediction.squeeze(dim=1), batch[-1])
        self.log("loss", loss, on_epoch=True)
        if self.config['MODEL']['Prediction_type'] == 'Regression':
            SSres = loss * batch[-1].shape[0]
            SStotal = torch.sum(torch.square(batch[-1] - torch.mean(batch[-1])))
            r2 = 1 - SSres / SStotal

            event_indicator = torch.ones(batch[-1].shape, dtype=torch.bool)
            risk = 1 / prediction.squeeze(dim=1)
            cindex = concordance_index_censored(event_indicator.cpu().detach().numpy(),
                                                event_time=batch[-1].cpu().detach().numpy(),
                                                estimate=risk.cpu().detach().numpy())
            self.log('cindex', cindex[0], on_epoch=True)
            self.log('r2', r2, on_epoch=True)

            if self.config['REGULARIZATION']['Label_smoothing']:
                loss = WeightedMSE(prediction.squeeze(dim=1), batch[-1], weights=self.weights,
                                   label_range=self.label_range)

        out = {'loss': loss, 'features': features.detach(), 'label': label}

        return out

    def training_epoch_end(self, training_step_outputs):
        if self.config['REGULARIZATION']['Feature_smoothing']:
            training_labels = torch.cat([out['label'] for i, out in enumerate(training_step_outputs)], dim=0)
            training_features = torch.cat([out['features'] for i, out in enumerate(training_step_outputs)], dim=0)
            self.FDS = FDS(feature_dim=training_features.shape[1], start_update=0, start_smooth=1, kernel='gaussian',
                           ks=7,
                           sigma=3)
            if self.current_epoch >= 0:
                self.FDS.update_last_epoch_stats(self.current_epoch)
                self.FDS.update_running_stats(training_features, training_labels, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        out = {}
        datadict, label = batch
        prediction = self.classifier(self.forward(datadict, label))
        val_loss = self.loss_fcn(prediction.squeeze(dim=1), batch[-1])
        self.log("val_loss", val_loss, on_epoch=True)
        if self.config['MODEL']['Prediction_type'] == 'Regression':
            SSres = val_loss * batch[-1].shape[0]
            SStotal = torch.sum(torch.square(batch[-1] - torch.mean(batch[-1])))
            r2 = 1 - SSres / SStotal

            event_indicator = torch.ones(batch[-1].shape, dtype=torch.bool)
            risk = 1 / prediction.squeeze(dim=1)
            cindex = concordance_index_censored(event_indicator.cpu(), event_time=batch[-1].cpu(), estimate=risk.cpu())
            self.log('cindex', cindex[0], on_epoch=True)
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
        prediction_labels = torch.cat([out['prediction'] for i, out in enumerate(validation_step_outputs)], dim=0)
        if self.config['MODEL']['Prediction_type'] == 'Regression':
            risk_score = 1 / prediction_labels
            fig = self.report.generate_cumulative_dynamic_auc(self.train_label, validation_labels, risk_score)
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
            self.log('worst_AE', worst_MAE)
            self.log('worst_AE', worst_MAE)
            if 'Anatomy' in self.config['DATA']['module']:
                grid_img = self.report.generate_report(worst_img)
                self.logger.experiment.add_image('validate_worst_case_img', grid_img, self.current_epoch)
            if 'Dose' in self.config['DATA']['module']:
                grid_dose = self.report.generate_report(worst_dose)
                self.logger.experiment.add_image('validate_worst_case_dose', grid_dose, self.current_epoch)
        else:
            fpr, tpr, _ = self.roc(prediction_labels, validation_labels)
            AUROC = self.accuracy(prediction_labels, validation_labels.int())
            specificity = self.specificity(prediction_labels, validation_labels.int())
            fig = self.report.plot_AUROC(tpr, fpr)
            self.logger.experiment.add_figure("AUC", fig, self.current_epoch)
            #
            self.log('Specificity', specificity)
            self.log('AUROC', AUROC)

    def test_step(self, batch, batch_idx):
        datadict, label = batch
        prediction = self.classifier(self.forward(datadict, label))
        test_loss = self.loss_fcn(prediction.squeeze(dim=1), batch[-1])
        return test_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return [optimizer], [scheduler]

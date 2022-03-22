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

        if config['MODEL']['Prediction_type'] == 'Regression':
            self.classifier = nn.Sequential(
                nn.LazyLinear(128),
                nn.LazyLinear(1)
            )
            self.loss_fcn = torch.nn.MSELoss()

        self.train_label = train_label

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
        loss = self.loss_fcn(prediction.squeeze(dim=1), batch[-1])
        self.log("loss", loss, on_epoch=True)
        if self.config['MODEL']['Prediction_type'] == 'Regression':
            report = PredictionReport(prediction, label)
            self.log('cindex', report.c_index()[0], on_epoch=True)
            self.log('r2', report.r2_index(), on_epoch=True)
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
        report = PredictionReport(prediction, label)
        if self.config['MODEL']['Prediction_type'] == 'Regression':
            self.log('cindex', report.c_index()[0], on_epoch=True)
            self.log('r2', report.r2_index(), on_epoch=True)
            MAE = torch.abs(prediction.flatten(0) - label)
            out['MAE'] = MAE
            if 'Dose' in self.config['DATA']['module']:
                out['dose'] = datadict['Dose']
            if 'Anatomy' in self.config['DATA']['module']:
                out['img'] = datadict['Anatomy']
        out['prediction'] = prediction.squeeze(dim=1)
        out['label'] = label

        return out

    def validation_epoch_end(self, validation_step_outputs):
        validation_labels = torch.cat([out['label'] for i, out in enumerate(validation_step_outputs)], dim=0)
        prediction_labels = torch.cat([out['prediction'] for i, out in enumerate(validation_step_outputs)], dim=0)
        report = PredictionReport(prediction_labels, validation_labels)
        if self.config['MODEL']['Prediction_type'] == 'Regression':
            fig = report.generate_cumulative_dynamic_auc(self.train_label)
            self.logger.experiment.add_figure("AUC", fig, self.current_epoch)
            worst_record = report.worst_case_show(self.config, validation_step_outputs)
            self.log('worst_AE', worst_record['worst_AE'])
            if 'Anatomy' in self.config['DATA']['module']:
                grid_img = report.generate_report(worst_record['worst_img'])
                self.logger.experiment.add_image('validate_worst_case_img', grid_img, self.current_epoch)
            if 'Dose' in self.config['DATA']['module']:
                grid_dose = report.generate_report(worst_record['worst_dose'])
                self.logger.experiment.add_image('validate_worst_case_dose', grid_dose, self.current_epoch)

        if self.config['MODEL']['Prediction_type'] == 'Classification':
            classification_out = report.classification_matrix()
            fig = report.plot_AUROC(classification_out['tpr'], classification_out['fpr'])
            self.logger.experiment.add_figure("AUC", fig, self.current_epoch)
            self.log('Specificity', classification_out['specificity'])
            self.log('AUROC', classification_out['accuracy'])

    def test_step(self, batch, batch_idx):
        datadict, label = batch
        prediction = self.classifier(self.forward(datadict, label))
        test_loss = self.loss_fcn(prediction.squeeze(dim=1), batch[-1])
        test_out = {}
        if self.config['MODEL']['Prediction_type'] == 'Regression':
            MAE = torch.abs(prediction.flatten(0) - label)
            test_out['MAE'] = MAE
            if 'Dose' in self.config['DATA']['module']:
                test_out['dose'] = datadict['Dose']
            if 'Anatomy' in self.config['DATA']['module']:
                test_out['img'] = datadict['Anatomy']
        test_out['prediction'] = prediction.squeeze(dim=1)
        test_out['label'] = label
        test_out['loss'] = test_loss
        return test_out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return [optimizer], [scheduler]

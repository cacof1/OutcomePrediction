import matplotlib.pyplot as plt
import torch
from torch import nn
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from Models.fds import FDS
from Losses.loss import WeightedMSE, CrossEntropy
from sksurv.metrics import concordance_index_censored


class MixModel(LightningModule):
    def __init__(self, module_dict, config, train_label, label_range=None, weights=None, loss_fcn=torch.nn.BCEWithLogitsLoss()):
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
        try:
            features = torch.cat([self.module_dict[key](datadict[key]) for key in self.module_dict.keys()], dim=1)
        except:
            print('model wrong!')
        if self.config['REGULARIZATION']['Feature_smoothing']:
            if self.training and self.current_epoch >= 1:
                features = self.FDS.smooth(features, labels, self.current_epoch)

        return features

    def training_step(self, batch, batch_idx):
        out = {}
        datadict, label = batch
        features = self.forward(datadict, label)
        prediction = self.classifier(features)
        print(prediction, label)
        loss = self.loss_fcn(prediction.squeeze(), batch[-1])
        out['features'] = features.detach()
        out['label'] = label
        out['prediction'] = prediction.detach()
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        if self.config['MODEL']['Prediction_type'] == 'Regression':
            if self.config['REGULARIZATION']['Label_smoothing']:
                loss = WeightedMSE(prediction.squeeze(dim=1), batch[-1], weights=self.weights,
                                   label_range=self.label_range)
            MAE = torch.abs(prediction.flatten(0) - label)
            out['MAE'] = MAE.detach()
            if 'Dose' in self.config['DATA']['module']:
                out['dose'] = datadict['Dose']
            if 'Anatomy' in self.config['DATA']['module']:
                out['img'] = datadict['Anatomy']
        # prefix = 'train_step_'
        # self.logger.report_step(prediction, label, self.global_step, prefix)
        # Label smoothing for Regression
        out['loss'] = loss
        return out

    def training_epoch_end(self, training_step_outputs):
        training_labels = torch.cat([out['label'] for i, out in enumerate(training_step_outputs)], dim=0)
        training_prediction = torch.cat([out['prediction'] for i, out in enumerate(training_step_outputs)], dim=0)

        prefix = 'train_epoch_'
        # self.logger.log_metrics({prefix+'loss': self.loss_fcn(training_prediction.squeeze(), training_labels)},
        # self.current_epoch)
        self.logger.report_epoch(training_prediction.squeeze(), training_labels, self.train_label,
                                 training_step_outputs, self.current_epoch, prefix)

        if self.config['REGULARIZATION']['Feature_smoothing']:
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
        val_loss = self.loss_fcn(prediction.squeeze(), batch[-1])
        self.log("val_loss", val_loss, on_step=False, on_epoch=True)
        # prefix = 'val_step_'
        # self.logger.report_step(prediction, label, batch_idx, prefix)
        # Finding worst case
        if self.config['MODEL']['Prediction_type'] == 'Regression':
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
        val_labels = torch.cat([out['label'] for i, out in enumerate(validation_step_outputs)], dim=0)
        val_prediction = torch.cat([out['prediction'] for i, out in enumerate(validation_step_outputs)], dim=0)
        prefix = 'val_epoch_'
        # self.logger.log_metrics({prefix + 'loss': self.loss_fcn(val_prediction.squeeze(), val_labels)},
        #                         self.current_epoch)
        self.logger.report_epoch(val_prediction.squeeze(), val_labels, self.train_label,
                                 validation_step_outputs, self.current_epoch, prefix)

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

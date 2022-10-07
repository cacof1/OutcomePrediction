import matplotlib.pyplot as plt
import torch
from torch import nn
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from Models.fds import FDS
from Losses.loss import WeightedMSE, CrossEntropy
from sksurv.metrics import concordance_index_censored
from monai.networks import nets

class MixModel(LightningModule):
    def __init__(self, module_dict, config, label_range=None, weights=None, loss_fcn=torch.nn.BCEWithLogitsLoss()):
        super().__init__()
        self.FDS = None
        self.module_dict = module_dict
        self.config = config
        self.label_range = label_range
        self.weights = weights

        self.loss_fcn   = getattr(torch.nn, self.config["MODEL"]["Loss_Function"])()
        self.activation = getattr(torch.nn, self.config["MODEL"]["Activation"])()
        # self.classifier = nets.Classifier((2, 2, 192), 1, (2, 4, 8), (2, 2, 2))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(512, 256, kernel_size=(5, 5), stride=(3, 3)),
            nn.Dropout(0.3),
            nn.Conv2d(256, 1, kernel_size=(4, 4), stride=(4, 4)),
            self.activation,
            nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )
        self.classifier.apply(self.weights_init)

    def forward(self, datadict):
        features = torch.cat([self.module_dict[key](datadict[key]) for key in self.module_dict.keys()], dim=1)
        # features = features.transpose(1, 3)
        prediction = self.classifier(features)
        return prediction

    def training_step(self, batch, batch_idx):
        out = {}
        datadict, label = batch
        prediction = self.forward(datadict)
        loss = self.loss_fcn(prediction.squeeze(), batch[-1])

        out['label'] = label
        out['prediction'] = prediction.detach()
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        if self.config['MODEL']['Prediction_type'] == 'Regression':
            if self.config['REGULARIZATION']['Label_smoothing']:
                loss = WeightedMSE(prediction.squeeze(dim=1), batch[-1], weights=self.weights, label_range=self.label_range)
            MAE = torch.abs(prediction.flatten(0) - label)
            out['MAE'] = MAE.detach()
            if 'Dose' in self.config['DATA']['module']: out['dose'] = datadict['Dose']
            if 'CT' in self.config['DATA']['module']: out['img'] = datadict['CT']

        out['loss'] = loss
        return out

    def training_epoch_end(self, training_step_outputs):
        training_labels = torch.cat([out['label'] for i, out in enumerate(training_step_outputs)], dim=0)
        training_prediction = torch.cat([out['prediction'] for i, out in enumerate(training_step_outputs)], dim=0)
        prefix = 'train_epoch_'
        self.logger.report_epoch(training_prediction.squeeze(), training_labels,training_step_outputs, self.current_epoch, prefix)
                                 
    def validation_step(self, batch, batch_idx):
        out = {}
        datadict, label = batch
        prediction = self.forward(datadict)
        val_loss = self.loss_fcn(prediction.squeeze(), batch[-1])
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, sync_dist=True)
        if self.config['MODEL']['Prediction_type'] == 'Regression':
            MAE = torch.abs(prediction.flatten(0) - label)
            out['MAE'] = MAE
            if 'Dose' in self.config['DATA']['module']: out['dose'] = datadict['Dose']
            if 'CT' in self.config['DATA']['module']: out['img'] = datadict['CT']
                
        out['prediction'] = prediction.squeeze(dim=1)
        out['label'] = label

        return out

    def validation_epoch_end(self, validation_step_outputs):
        val_labels = torch.cat([out['label'] for i, out in enumerate(validation_step_outputs)], dim=0)
        val_prediction = torch.cat([out['prediction'] for i, out in enumerate(validation_step_outputs)], dim=0)
        prefix = 'val_epoch_'
        self.logger.report_epoch(val_prediction.squeeze(), val_labels, validation_step_outputs, self.current_epoch, prefix)

    def test_step(self, batch, batch_idx):
        datadict, label = batch
        prediction = self.forward(datadict, label)
        test_loss = self.loss_fcn(prediction.squeeze(dim=1), batch[-1])
        test_out = {}
        if self.config['MODEL']['Prediction_type'] == 'Regression':
            MAE = torch.abs(prediction.flatten(0) - label)
            test_out['MAE'] = MAE
            if 'Dose' in self.config['DATA']['module']:
                test_out['dose'] = datadict['Dose']
            if 'CT' in self.config['DATA']['module']:
                test_out['img'] = datadict['CT']
        test_out['prediction'] = prediction.squeeze(dim=1)
        test_out['label'] = label
        test_out['loss'] = test_loss
        return test_out

    def weights_init(self, m):
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)

    def weights_reset(self, m):
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return [optimizer], [scheduler]

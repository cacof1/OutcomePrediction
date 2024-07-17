import matplotlib.pyplot as plt
import torch
import copy
from torch import nn
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from Losses.loss import WeightedMSE, CrossEntropy
from sksurv.metrics import concordance_index_censored
from monai.networks import nets

class MixModel(LightningModule):
    def __init__(self, module_dict, config, loss_fcn=torch.nn.BCEWithLogitsLoss()):
        super().__init__()
        self.module_dict = module_dict
        self.config      = config
        self.loss_fcn    = getattr(torch.nn, self.config["MODEL"]["loss_function"])(pos_weight=torch.tensor(1.21))  # TODO: Why 1.21??, Doesn't work with CrossEntropyLoss
        self.activation  = getattr(torch.nn, self.config["MODEL"]["activation"])()
        self.classifier  = nn.Sequential(
            nn.Linear(124, 120),
            nn.Dropout(0.3),
            nn.Linear(120, 40),
            nn.Dropout(0.3),
            nn.Linear(40, config['DATA']['n_classes']),
            #self.activation
        )
        self.classifier.apply(self.weights_init)

        self.training_outputs = []
        self.validation_outputs = []

    def forward(self, data_dict):
        features   = torch.cat([self.module_dict[key](data_dict[key]) for key in self.module_dict.keys()
                                if key in data_dict.keys()], dim=1)
        prediction = self.classifier(features)
        return prediction

    def training_step(self, batch, batch_idx):
        out = {}
        data_dict, label = batch
        prediction = self.forward(data_dict)
        loss = self.loss_fcn(prediction.squeeze(dim=1), label)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        MAE = torch.abs(prediction.flatten(0) - label)
        out['MAE'] = MAE.detach()
        out = copy.deepcopy(data_dict)
        out['prediction'] = prediction.detach()
        out['label'] = label
        out['loss'] = loss
        self.training_outputs.append(out)
        return out

    def on_train_epoch_end(self):
        labels = torch.cat([out['label'] for i, out in enumerate(self.training_outputs)], dim=0)
        prediction = torch.cat([out['prediction'] for i, out in enumerate(self.training_outputs)], dim=0)
        # self.logger.report_epoch(prediction, labels, self.training_outputs,self.current_epoch, 'train_epoch_')
        self.training_outputs.clear()
                                 
    def validation_step(self, batch, batch_idx):
        out = {}
        data_dict, label = batch
        prediction = self.forward(data_dict)
        loss = self.loss_fcn(prediction.squeeze(dim=1), label)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        MAE = torch.abs(prediction.flatten(0) - label)
        out['MAE'] = MAE
        out = copy.deepcopy(data_dict)
        out['prediction'] = prediction
        out['label'] = label
        out['loss'] = loss
        self.validation_outputs.append(out)
        return out

    def on_validation_epoch_end(self):
        labels = torch.cat([out['label'] for i, out in enumerate(self.validation_outputs)], dim=0)
        prediction = torch.cat([out['prediction'] for i, out in enumerate(self.validation_outputs)], dim=0)
        # self.logger.report_epoch(prediction.squeeze(), labels, self.validation_outputs, self.current_epoch,'val_epoch_')
        self.validation_outputs.clear()

    def test_step(self, batch, batch_idx):
        data_dict, label = batch
        prediction = self.forward(data_dict)
        loss = self.loss_fcn(prediction.squeeze(dim=1), label)
        out = {}
        MAE = torch.abs(prediction.flatten(0) - label)
        out['MAE'] = MAE
        out = copy.deepcopy(data_dict)
        out['prediction'] = prediction.squeeze(dim=1)
        out['label'] = label
        out['loss'] = loss                
        return out

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

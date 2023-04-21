import matplotlib.pyplot as plt
import torch
import numpy as np
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
        out_feat     = np.sum([model.out_feat for model in module_dict.values()])
        self.config      = config
        self.loss_fcn    = getattr(torch.nn, self.config["MODEL"]["Loss_Function"])(pos_weight=torch.tensor(1.18))
        self.activation  = getattr(torch.nn, self.config["MODEL"]["Activation"])()
        self.classifier  = nn.Sequential(
            nn.Linear(out_feat, 256),
            nn.Dropout(0.05),
            nn.Linear(256, 120),
            nn.Dropout(0.05),
            nn.Linear(120, 40),
            nn.Dropout(0.05),
            nn.Linear(40, config['DATA']['n_classes']),
            self.activation
        )

    def forward(self, data_dict):
        features   = torch.cat([self.module_dict[key](data_dict[key]) for key in self.module_dict.keys()], dim=1)
        prediction = self.classifier(features)
        return prediction

    def training_step(self, batch, batch_idx):
        data_dict, censor_status, label = batch
        prediction = self.forward(data_dict).squeeze(dim=1)
        loss = self.loss_fcn(prediction, label)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        MAE = torch.abs(prediction - label)
        out = copy.deepcopy(data_dict)
        out['MAE']   = MAE.detach()
        out['prediction'] = prediction.detach()
        out['label'] = label
        out['censor_status'] = censor_status
        out['loss']  = loss
        return out

    def training_epoch_end(self, step_outputs):
        labels = torch.cat([out['label'] for i, out in enumerate(step_outputs)], dim=0)
        censor_status = torch.cat([out['censor_status'] for i, out in enumerate(step_outputs)], dim=0)
        prediction = torch.cat([out['prediction'] for i, out in enumerate(step_outputs)], dim=0)
        self.logger.report_epoch(prediction, censor_status, labels, step_outputs,self.current_epoch, 'train_epoch_')
        # with open(self.logger.log_dir + "/train_record.ini", "a") as toml_file:
        #     toml_file.write('\n')
        #     toml_file.write('label_epoch_' + str(self.current_epoch) + ':\n')
        #     toml_file.write(str(labels[1]))
        #     toml_file.write('\n')
        #     toml_file.write('censor_epoch_' + str(self.current_epoch) + ':\n')
        #     toml_file.write(str(labels[0]))
        #     toml_file.write('\n')
        #     toml_file.write('prediction_epoch_' + str(self.current_epoch) + ':\n')
        #     toml_file.write(str(prediction))
        #     toml_file.write('\n')
                         
    def validation_step(self, batch, batch_idx):
        data_dict, censor_status, label = batch
        prediction = self.forward(data_dict).squeeze(dim=1)
        loss = self.loss_fcn(prediction, label)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        MAE = torch.abs(prediction - label)
        out = copy.deepcopy(data_dict)
        out['MAE'] = MAE
        out['prediction'] = prediction
        out['censor_status'] = censor_status
        out['label'] = label
        out['loss'] = loss        
        return out

    def validation_epoch_end(self, step_outputs):
        labels = torch.cat([out['label'] for i, out in enumerate(step_outputs)], dim=0)
        censor_status = torch.cat([out['censor_status'] for i, out in enumerate(step_outputs)], dim=0)
        prediction = torch.cat([out['prediction'] for i, out in enumerate(step_outputs)], dim=0)
        self.logger.report_epoch(prediction, censor_status, labels, step_outputs, self.current_epoch, 'val_epoch_')
        # with open(self.logger.log_dir + "/val_record.ini", "a") as toml_file:
        #     toml_file.write('\n')
        #     toml_file.write('label_epoch_' + str(self.current_epoch) + ':\n')
        #     toml_file.write(str(labels[1]))
        #     toml_file.write('\n')
        #     toml_file.write('censor_epoch_' + str(self.current_epoch) + ':\n')
        #     toml_file.write(str(labels[0]))
        #     toml_file.write('\n')
        #     toml_file.write('prediction_epoch_' + str(self.current_epoch) + ':\n')
        #     toml_file.write(str(prediction))
        #     toml_file.write('\n')

    def test_step(self, batch, batch_idx):
        data_dict, censor_status, label = batch
        prediction = self.forward(data_dict).squeeze(dim=1)
        loss = self.loss_fcn(prediction, label)
        MAE = torch.abs(prediction - label)
        out = copy.deepcopy(data_dict)
        out['MAE'] = MAE
        out['prediction'] = prediction
        out['label'] = label
        out['censor_status'] = censor_status
        out['loss'] = loss
        return out

    def weights_init(self, m):
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)

    def weights_reset(self, m):
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()
            
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return [optimizer], [scheduler]

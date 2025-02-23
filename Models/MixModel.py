from typing import Any, Dict, Tuple
import matplotlib.pyplot as plt
import torch
import copy
import pandas as pd
from torch import nn
import torch.distributed as dist
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryF1Score
from torchmetrics.regression import MeanAbsoluteError, R2Score, MeanAbsolutePercentageError, MeanSquaredError
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from torcheval.metrics.aggregation.auc import AUC
from torcheval.metrics.toolkit import sync_and_compute
from torch.optim.lr_scheduler import LambdaLR, SequentialLR, StepLR

from Losses.loss import WeightedMSE, CrossEntropy, MaskedMSELoss
from sksurv.metrics import concordance_index_censored
from monai.networks import nets


def get_parameter_mean(model):
    weights = [p.data for p in model.parameters() if p.requires_grad]
    all_weights = torch.cat([w.flatten() for w in weights])
    mean_weight = all_weights.mean().item()
    return mean_weight


class MixModel(LightningModule):
    def __init__(self, module_dict, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.module_dict = module_dict
        self.config = config
        # self.loss_fcn = getattr(torch.nn, self.config["MODEL"]["loss_function"])(pos_weight=torch.tensor(1.21))  # TODO: Why 1.21??, Doesn't work with CrossEntropyLoss
        self.loss_fcns = [getattr(torch.nn, elem)(reduction="mean") for elem in self.config["MODEL"]["loss_functions"]]
        self.loss_fcns = [elem if elem is not torch.nn.MSELoss else MaskedMSELoss for elem in self.loss_fcns]
        self.activations = [getattr(torch.nn, elem)() for elem in self.config["MODEL"]["activations"]]
        self.loss_weights_2 = (torch.ones(len(self.loss_fcns))*self.config["MODEL"]["loss_weights"]
                               if type(self.config["MODEL"]["loss_weights"]) is not list
                               else self.config["MODEL"]["loss_weights"])
        loss_weights = config["MODEL"]["loss_weights"]
        self.loss_weights = torch.tensor(
            loss_weights if isinstance(loss_weights, list) else [loss_weights] * len(self.loss_fcns))
        layers = ([config['MODEL']['classifier_in']] + config['MODEL']['classifier_config'] +
                  [config['DATA']['n_classes']])
        self.classifier = nn.Sequential()
        if config['MODEL']['backbone'] != 'efficientnet':
            for i in range(len(layers)-1):
                self.classifier += nn.Sequential(
                    nn.Linear(layers[i], layers[i+1]),
                    nn.Dropout(config['MODEL']['dropout_prob'])
                )
        else:
            self.classifier += nn.Sequential(nn.Identity())
        self.classifier.apply(self.weights_init)
        self.survival_prediction_mode = config['MODEL']['modes'][0]

        # self.classifier = nn.Sequential(
        #     nn.Linear(config['MODEL']['classifier_in'], 120),
        #     nn.Dropout(config['MODEL']['dropout_prob']),
        #     nn.Linear(120, 40),
        #     nn.Dropout(config['MODEL']['dropout_prob']),
        #     nn.Linear(40, config['DATA']['n_classes']),
        #     # self.activation
        # )
        # self.classifier.apply(self.weights_init)
        # self.survival_prediction_mode = config['MODEL']['modes'][0]

        if self.survival_prediction_mode == 'classification':
            self.train_accuracy = BinaryAccuracy()
            self.train_auc = BinaryAUROC()
            self.train_f1score = BinaryF1Score()
            self.validation_accuracy = BinaryAccuracy()
            self.validation_auc = BinaryAUROC()
            self.validation_f1score = BinaryF1Score()

        if self.survival_prediction_mode == 'regression':
            self.train_mae = MeanAbsoluteError()
            self.train_mape = MeanAbsolutePercentageError()
            # self.train_r2 = R2Score()
            self.validation_mae = MeanAbsoluteError()
            self.validation_mape = MeanAbsolutePercentageError()
            # self.validation_r2 = R2Score()

    def forward(self, data_dict):
        features = torch.cat([self.module_dict[k](data_dict[k]) for k in self.module_dict if k in data_dict], dim=1)
        prediction = self.classifier(features)
        return prediction

    def compute_loss_and_metrics(self, prediction: torch.Tensor, label: torch.Tensor, mode: str, stage: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        mask = ~torch.isnan(label)
        loss = torch.tensor(0.0, device=prediction.device)
        for i in range(label.shape[1]):
            if mask[:, i].any():
                loss += self.loss_weights[i] * self.loss_fcns[i](prediction[mask[:, i], i], label[mask[:, i], i])
        loss = loss / mask.any(dim=0).sum()

        survival_prediction = prediction[mask[:, 0], 0]
        survival_label = label[mask[:, 0], 0]
        pred_final = self.activations[0](survival_prediction.detach())

        if survival_label.numel() > 0:
            if mode == 'classification':
                if stage == 'train':
                    self.train_accuracy(pred_final, survival_label)
                    self.train_auc(pred_final, survival_label)
                    self.train_f1score(pred_final, survival_label)
                elif stage == 'val':
                    self.validation_accuracy(pred_final, survival_label)
                    self.validation_auc(pred_final, survival_label)
                    self.validation_f1score(pred_final, survival_label)
            elif mode == 'regression':
                if stage == 'train':
                    self.train_mae(pred_final, survival_label)
                    self.train_mape(pred_final, survival_label)
                elif stage == 'val':
                    self.validation_mae(pred_final, survival_label)
                    self.validation_mape(pred_final, survival_label)
        return loss, {'prediction': prediction, 'label': label}

    def training_step(self, batch, batch_idx):
        data_dict, label = batch[:2] if 'censor_label' in self.config['DATA'] else batch
        prediction = self.forward(data_dict)
        loss, metrics = self.compute_loss_and_metrics(prediction, label, self.survival_prediction_mode, stage='train' if self.training else 'val')

        if batch_idx == 0:
            lr = self.trainer.lr_scheduler_configs[0].scheduler.optimizer.param_groups[0]['lr']
            print(f"IMAGE ID 1 MEAN: {data_dict['Image'].mean()}, LABEL: {label.mean()}, LEARNING RATE: {lr}, LOSS: {loss.mean()}")

        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return {**copy.deepcopy(data_dict), **metrics, 'loss': loss}

    def on_train_epoch_end(self):
        if self.survival_prediction_mode == 'classification':
            self.log('train_accuracy_epoch', self.train_accuracy, on_step=False, on_epoch=True, sync_dist=True,
                     prog_bar=True)
            self.log("train_auc_epoch", self.train_auc, on_step=False, on_epoch=True, sync_dist=True,
                     prog_bar=False)
            self.log('train_f1score_epoch', self.train_f1score, on_step=False, on_epoch=True, sync_dist=True,
                     prog_bar=False)
        elif self.survival_prediction_mode == 'regression':
            self.log('train_mae_epoch', self.train_mae, on_step=False, on_epoch=True, sync_dist=True,
                     prog_bar=True)
            self.log("train_mape_epoch", self.train_mape, on_step=False, on_epoch=True, sync_dist=True,
                     prog_bar=False)
            # self.log('train_r2_epoch', self.train_r2, on_step=False, on_epoch=True, sync_dist=True,
            #          prog_bar=False)

    def validation_step(self, batch, batch_idx):
        data_dict, label = batch[:2] if 'censor_label' in self.config['DATA'] else batch
        prediction = self.forward(data_dict)
        loss, metrics = self.compute_loss_and_metrics(prediction, label, self.survival_prediction_mode, stage='train' if self.training else 'val')
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return {**copy.deepcopy(data_dict), **metrics, 'loss': loss}

    def on_validation_epoch_end(self):
        if self.survival_prediction_mode == 'classification':
            self.log('validation_accuracy_epoch', self.validation_accuracy, on_step=False, on_epoch=True,
                     sync_dist=True, prog_bar=True)
            self.log('validation_auc_epoch', self.validation_auc, on_step=False, on_epoch=True, sync_dist=True,
                     prog_bar=False)
            self.log('validation_f1score_epoch', self.validation_f1score, on_step=False, on_epoch=True,
                     sync_dist=True, prog_bar=False)
        elif self.survival_prediction_mode == 'regression':
            self.log('validation_mae_epoch', self.validation_mae, on_step=False, on_epoch=True,
                     sync_dist=True, prog_bar=True)
            self.log('validation_mape_epoch', self.validation_mape, on_step=False, on_epoch=True, sync_dist=True,
                     prog_bar=False)
            # self.log('validation_r2_epoch', self.validation_r2, on_step=False, on_epoch=True,
            #          sync_dist=True, prog_bar=False)

    def test_step(self, batch, batch_idx):
        data_dict, label = batch[:2] if 'censor_label' in self.config['DATA'] else batch
        prediction = self.forward(data_dict)
        loss, metrics = self.compute_loss_and_metrics(prediction, label, self.survival_prediction_mode, stage='test')
        return {**copy.deepcopy(data_dict), **metrics, 'loss': loss}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        data_dict = batch[0]
        prediction = self.forward(data_dict)
        prediction_final = torch.cat(
            [self.activations[i](prediction[:, i])[:, None] for i in range(prediction.shape[1])], dim=1)
        return prediction_final, *batch[1:]

    def weights_init(self, m):
        if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight.data)

    def weights_reset(self, m):
        if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Linear)):
            m.reset_parameters()
            
    def configure_optimizers(self):
        def lr_lambda(epoch):
            warmup_epochs = self.config['MODEL']['lr_warmup_epochs']
            return (epoch + 1) / (warmup_epochs + 1) if epoch < warmup_epochs else 1.0

        opt_cls = getattr(torch.optim, self.config['MODEL']['optimizer'], torch.optim.Adam)
        optimizer = opt_cls(self.parameters(), lr=self.config['MODEL']['learning_rate'])
        #     optimizer = eval(f'torch.optim.{self.config["MODEL"]["optimizer"]}')
        #     optimizer = optimizer(self.parameters(), lr=self.config['MODEL']['learning_rate'])
        # else:
        #     optimizer = torch.optim.Adam(self.parameters(), lr=self.config['MODEL']['learning_rate'])

        warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        decay_scheduler = StepLR(optimizer, step_size=self.config['MODEL']['lr_step_size'],
                                 gamma=self.config['MODEL']['lr_gamma'])
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, decay_scheduler],
                                 milestones=[self.config['MODEL']['lr_warmup_epochs']])
        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch', 'frequency': 1}}


    # def configure_optimizers(self):
    #     if 'optimizer' in self.config['MODEL']:
    #         optimizer = eval(f'torch.optim.{self.config["MODEL"]["optimizer"]}')
    #         optimizer = optimizer(self.parameters(), lr=self.config['MODEL']['learning_rate'])
    #     else:
    #         optimizer = torch.optim.Adam(self.parameters(), lr=self.config['MODEL']['learning_rate'])
    #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config['MODEL']['lr_step_size'],
    #                                                 gamma=self.config['MODEL']['lr_gamma'])
    #     return [optimizer], [scheduler]



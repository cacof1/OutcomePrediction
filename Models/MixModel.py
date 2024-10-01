from typing import Any

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

from Losses.loss import WeightedMSE, CrossEntropy
from sksurv.metrics import concordance_index_censored
from monai.networks import nets


class MixModel(LightningModule):
    def __init__(self, module_dict, config, loss_fcn=torch.nn.BCEWithLogitsLoss()):
        super().__init__()
        self.module_dict = module_dict
        self.config = config
        # self.loss_fcn = getattr(torch.nn, self.config["MODEL"]["loss_function"])(pos_weight=torch.tensor(1.21))  # TODO: Why 1.21??, Doesn't work with CrossEntropyLoss
        self.loss_fcns = [getattr(torch.nn, elem)() for elem in self.config["MODEL"]["loss_functions"]]
        self.activations = [getattr(torch.nn, elem)() for elem in self.config["MODEL"]["activations"]]
        self.loss_weights = (torch.ones(len(self.loss_fcns))*self.config["MODEL"]["loss_weights"]
                             if type(self.config["MODEL"]["loss_weights"]) is not list
                             else [getattr(torch.nn, elem)() for elem in self.config["MODEL"]["loss_weights"]])
        self.classifier = nn.Sequential(
            nn.Linear(config['MODEL']['classifier_in'], 120),
            nn.Dropout(config['MODEL']['dropout_prob']),
            nn.Linear(120, 40),
            nn.Dropout(config['MODEL']['dropout_prob']),
            nn.Linear(40, config['DATA']['n_classes']),
            # self.activation
        )
        self.classifier.apply(self.weights_init)
        self.survival_prediction_mode = config['MODEL']['modes'][0]

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

        self.training_outputs = []
        self.validation_outputs = []

    def forward(self, data_dict):
        features = torch.cat([self.module_dict[key](data_dict[key]) for key in self.module_dict.keys()
                             if key in data_dict.keys()], dim=1)
        prediction = self.classifier(features)
        return prediction

    def training_step(self, batch, batch_idx):
        out = {}
        data_dict, label = batch if 'censor_label' not in self.config['DATA'].keys() else batch[:2]
        prediction = self.forward(data_dict)
        survival_prediction = prediction[:, 0]
        survival_label = label[:, 0]
        loss = self.loss_weights[0] * self.loss_fcns[0](prediction[:, 0], label[:, 0])
        for i in range(1, label.shape[1]):
            loss += self.loss_weights[i] * self.loss_fcns[i](prediction[:, i], label[:, i])
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        prediction_final = self.activations[0](survival_prediction.detach())
        if self.survival_prediction_mode == 'classification':
            self.train_accuracy(prediction_final, survival_label)
            self.train_auc(prediction_final, survival_label)
            self.train_f1score(prediction_final, survival_label)
        elif self.survival_prediction_mode == 'regression':
            self.train_mae(prediction_final, survival_label)
            self.train_mape(prediction_final, survival_label)
            # self.train_r2(prediction_final, survival_label)
        # self.log('train_acc_step', self.train_accuracy, sync_dist=True)
        MAE = torch.abs(prediction_final - survival_label)
        out['MAE'] = MAE.detach()
        out = copy.deepcopy(data_dict)
        out['prediction'] = torch.concat(
            [self.activations[i](prediction[:, i])[:, None] for i in range(label.shape[1])], dim=1)
        out['label'] = label
        out['loss'] = loss
        # train_results = torch.cat([self.activation(prediction.detach()), label[:, None]], dim=1)
        # train_results_list = [torch.zeros_like(train_results) for _ in range(dist.get_world_size())]
        # dist.all_gather(train_results_list, train_results)
        # if len(self.training_outputs) > 0:
        #     self.training_outputs[0] = torch.cat(
        #         [self.training_outputs[0], torch.cat(train_results_list, dim=0)], dim=0)
        # else:
        #     self.training_outputs.append(torch.cat(train_results_list, dim=0))
        return out

    def on_train_epoch_end(self):
        # label = torch.cat([out['label'] for i, out in enumerate(self.training_outputs)], dim=0)
        # prediction = torch.cat([out['prediction'] for i, out in enumerate(self.training_outputs)], dim=0)
        # self.log("raw_train_results", torch.cat([labels[:, None], prediction], dim=1), sync_dist=True)
        # self.logger.report_epoch(prediction, labels, self.training_outputs,self.current_epoch, 'train_epoch_')
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
        # if self.global_rank == 0:
        #     results = pd.DataFrame(self.training_outputs[0].cpu(), columns=['Prediction', 'Target'])
        #     print(results)
        #     print(accuracy_score(results['Target'].values, results['Prediction'].values >= 0.5))
        #     # print(roc_auc_score(results['Target'], results['Prediction']))
        #     print(f1_score(results['Target'].values, results['Prediction'].values >= 0.5, average='macro'))
        self.training_outputs.clear()
                                 
    def validation_step(self, batch, batch_idx):
        out = {}
        data_dict, label = batch if 'censor_label' not in self.config['DATA'].keys() else batch[:2]
        prediction = self.forward(data_dict)
        survival_prediction = prediction[:, 0]
        survival_label = label[:, 0]
        loss = self.loss_weights[0] * self.loss_fcns[0](prediction[:, 0], label[:, 0])
        for i in range(1, label.shape[1]):
            loss += self.loss_weights[i] * self.loss_fcns[i](prediction[:, i], label[:, i])
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        prediction_final = self.activations[0](survival_prediction.detach())
        if self.survival_prediction_mode == 'classification':
            self.validation_accuracy(prediction_final, survival_label)
            self.validation_auc(prediction_final, survival_label)
            self.validation_f1score(prediction_final, survival_label)
        elif self.survival_prediction_mode == 'regression':
            self.validation_mae(prediction_final, survival_label)
            self.validation_mape(prediction_final, survival_label)
            # self.validation_r2(prediction_final, survival_label)
        MAE = torch.abs(prediction_final - survival_label)
        out['MAE'] = MAE
        out = copy.deepcopy(data_dict)
        out['prediction'] = torch.concat(
            [self.activations[i](prediction[:, i])[:, None] for i in range(label.shape[1])], dim=1)
        out['label'] = label
        out['loss'] = loss
        return out

    def on_validation_epoch_end(self):
        # labels = torch.cat([out['label'] for i, out in enumerate(self.validation_outputs)], dim=0)
        # prediction = torch.cat([out['prediction'] for i, out in enumerate(self.validation_outputs)], dim=0)
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
        self.validation_outputs.clear()

    def test_step(self, batch, batch_idx):
        data_dict, label = batch if 'censor_label' not in self.config['DATA'].keys() else batch[:2]
        prediction = self.forward(data_dict)
        survival_prediction = prediction[:, 0]
        survival_label = label[:, 0]
        prediction_final = self.activations[0](survival_prediction.detach())
        loss = self.loss_weights[0] * self.loss_fcns[0](prediction[:, 0], label[:, 0])
        for i in range(1, label.shape[1]):
            loss += self.loss_weights[i] * self.loss_fcns[i](prediction[:, i], label[:, i])
        out = {}
        MAE = torch.abs(prediction_final - survival_label)
        out['MAE'] = MAE
        out = copy.deepcopy(data_dict)
        out['prediction'] = torch.concat(
            [self.activations[i](prediction[:, i])[:, None] for i in range(label.shape[1])], dim=1)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['MODEL']['learning_rate'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.config['MODEL']['lr_step_size'],
                                                    gamma=self.config['MODEL']['lr_gamma'])
        return [optimizer], [scheduler]

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        data_dict = batch[0]
        prediction = self.forward(data_dict)
        prediction_final = torch.concat(
            [self.activations[i](prediction[:, i])[:, None] for i in range(prediction.shape[1])], dim=1)
        return prediction_final, *batch[1:]

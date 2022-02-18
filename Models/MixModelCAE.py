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
from Losses.loss import WeightedMSE
## Models
from Models.Linear import Linear
from Models.Classifier2D import Classifier2D
from Models.Classifier3D import Classifier3D
from Models.TransformerEncoder import PositionEncoding, PatchEmbedding, TransformerBlock
from Models.fds import FDS


class MixModelCAE(LightningModule):
    def __init__(self, module_dict, img_sizes, patch_size, embed_dim, in_channels, num_layers=3, weights=None, label_range=None,
                 loss_fcn=torch.nn.BCEWithLogitsLoss()):
        super().__init__()
        self.module_dict = module_dict
        self.weights = weights
        self.label_range = label_range
        ## define backbone
        backbone = torchvision.models.resnet18(pretrained=True)
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.feature_extractor.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.linear1 = nn.LazyLinear(256)
        self.FDS = FDS(feature_dim=1024, start_update=0, start_smooth=1, kernel='gaussian', ks=7, sigma=3)

        self.pe = PositionEncoding(img_size=img_sizes, patch_size=patch_size, in_channel=in_channels,
                                   embed_dim=embed_dim, dropout=0.5, img_dim=2, iftoken=False)

        self.transformers = nn.ModuleList(
            [TransformerBlock(num_heads=16, embed_dim=embed_dim, mlp_dim=128, dropout=0.5) for _ in range(num_layers)])
        self.pool_top = nn.MaxPool2d(4)

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
        loss = loss / (i+1)
        return loss

    def convert2d(self, x):
        y = x.repeat(1, 3, 1, 1)
        features = self.feature_extractor(y)
        features = features.permute(2, 3, 0, 1)
        features = self.linear1(features)
        return features

    def forward(self, datadict, labels):
        # features = torch.cat([self.module_dict[key](datadict[key]) for key in self.module_dict.keys()], dim=1)
        # For transformer
        for key in self.module_dict.keys():
            if "Dose" == key or "Anatomy" == key:
                x = datadict[key]
                features = torch.cat([self.convert2d(b.transpose(0, 1)) for i, b in enumerate(x)], dim=0)
                # features_pe = self.pe(features)
                features = features.permute(0, 2, 3, 1).flatten(2)
                for transformer in self.transformers:
                    x = transformer(features)

                x = self.pool_top(x)
                features = x.flatten(start_dim=1)
                if self.training and self.current_epoch >= 1:
                    features = self.FDS.smooth(features, labels, self.current_epoch)

            if "Clinical" == key:
                if flg == 0:
                    features = self.module_dict[key](datadict[key])
                    flg = 1
                else:
                    features = torch.cat((features, self.module_dict[key](datadict[key])), dim=1)

        out = {'features': features, 'prediction': self.classifier(features)}

        return out

    def training_step(self, batch, batch_idx):
        datadict, label = batch
        forward_cal = self.forward(datadict, label)
        prediction = forward_cal['prediction']
        print(prediction, label)
        # loss = self.loss_fcn(prediction.squeeze(dim=1), batch[-1])
        loss = self.WeightedMSE(prediction.squeeze(dim=1), batch[-1])
        self.log("loss", loss, on_epoch=True)
        out = {'loss': loss, 'features': forward_cal['features'], 'label': label}
        return out

    def training_epoch_end(self, training_step_outputs):
        # for i, out in enumerate(training_step_outputs):
        #     print(out['features'])
        training_features = torch.cat([out['features'] for i, out in enumerate(training_step_outputs)], dim=0)
        training_labels = torch.cat([out['label'] for i, out in enumerate(training_step_outputs)], dim=0)
        if self.current_epoch >= 0:
            self.FDS.update_last_epoch_stats(self.current_epoch)
            self.FDS.update_running_stats(training_features, training_labels, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        datadict, label = batch
        forward_cal = self.forward(datadict, label)
        prediction = forward_cal['prediction']
        val_loss = self.loss_fcn(prediction.squeeze(dim=1), batch[-1])
        self.log("val_loss", val_loss, on_epoch=True)
        MAE = torch.abs(prediction.flatten(0) - label)
        out = {'MAE': MAE, 'img': datadict['Anatomy']}
        return out

    @staticmethod
    def generate_report(img):
        # img_batch = batch[0]['Anatomy']
        img_batch = img.view(img.shape[0] * img.shape[1], *[1, img.shape[2], img.shape[3]])
        grid = torchvision.utils.make_grid(img_batch)
        return grid

    def validation_epoch_end(self, validation_step_outputs):
        worst_MAE = 0
        for i, data in enumerate(validation_step_outputs):
            loss = data['MAE']
            idx = torch.argmax(loss)
            if loss[idx] > worst_MAE:
                worst_img = data['img'][idx]
                worst_MAE = loss[idx]
        self.log('worst_MAE', worst_MAE)
        grid = self.generate_report(worst_img)
        self.logger.experiment.add_image('validate_worst_case_img', grid, self.current_epoch)

    def test_step(self, batch, batch_idx):
        datadict, label = batch
        forward_cal = self.forward(datadict, label)
        prediction = forward_cal['prediction']
        test_loss = self.loss_fcn(prediction.squeeze(dim=1), batch[-1])
        print('test_prediction:', prediction, label)
        print('test_loss:', test_loss)
        return test_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return [optimizer], [scheduler]

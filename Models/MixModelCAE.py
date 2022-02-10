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

## Models
from Models.Linear import Linear
from Models.Classifier2D import Classifier2D
from Models.Classifier3D import Classifier3D
from Models.TransformerEncoder import PositionEncoding, PatchEmbedding, TransformerBlock


class MixModelCAE(LightningModule):
    def __init__(self, module_dict, img_sizes, patch_size, embed_dim, in_channels, num_layers=3,
                 loss_fcn=torch.nn.BCEWithLogitsLoss()):
        super().__init__()
        self.module_dict = module_dict
        ## define backbone
        backbone = torchvision.models.resnet18(pretrained=True)
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.feature_extractor.eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.linear1 = nn.LazyLinear(256)

        self.pe = PositionEncoding(img_size=img_sizes, patch_size=patch_size, in_channel=in_channels, embed_dim=embed_dim, dropout=0.5, img_dim=2, iftoken=False)

        self.transformers = nn.ModuleList(
            [TransformerBlock(num_heads=16, embed_dim=embed_dim, mlp_dim=128, dropout=0.5) for _ in range(num_layers)])
        self.pool_top = nn.MaxPool2d(4)

        self.classifier = nn.Sequential(
            nn.LazyLinear(128),
            nn.LazyLinear(1),
            nn.ReLU()
        )
        self.accuracy = torchmetrics.AUC(reorder=True)
        self.loss_fcn = torch.nn.MSELoss()  # loss_fcn

    def convert2D(self, x):
        y = x.repeat(1, 3, 1, 1)
        features = self.feature_extractor(y)
        features = features.permute(2, 3, 0, 1)
        features = self.linear1(features)
        return features

    def forward(self, datadict):
        # features = torch.cat([self.module_dict[key](datadict[key]) for key in self.module_dict.keys()], dim=1)
        # For transformer
        for key in self.module_dict.keys():
            if "Dose" == key or "Anatomy" == key:
                x = datadict[key]
                features = torch.cat([self.convert2D(b.transpose(0, 1)) for i, b in enumerate(x)], dim=0)
                features_pe = self.pe(features)

                for transformer in self.transformers:
                    x = transformer(features_pe)

                x = self.pool_top(x)
                features = x.flatten(start_dim=1)

            if "Clinical" == key:
                if flg == 0:
                    features = self.module_dict[key](datadict[key])
                    flg = 1
                else:
                    features = torch.cat((features, self.module_dict[key](datadict[key])), dim=1)

        return self.classifier(features)

    def training_step(self, batch, batch_idx):
        datadict, label = batch
        prediction = self.forward(datadict)
        print(prediction, label)
        loss = self.loss_fcn(prediction.squeeze(dim=1), batch[-1])
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        datadict, label = batch
        prediction = self.forward(datadict)
        val_loss = self.loss_fcn(prediction.squeeze(dim=1), batch[-1])
        self.log("val_loss", val_loss)
        print('val_prediction:', prediction, label)
        print('val_loss:', val_loss)
        return val_loss

    def test_step(self, batch, batch_idx):
        datadict, label = batch
        prediction = self.forward(datadict)
        test_loss = self.loss_fcn(prediction.squeeze(dim=1), batch[-1])
        print('test_prediction:', prediction, label)
        print('test_loss:', test_loss)
        return test_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return [optimizer], [scheduler]

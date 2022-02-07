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
from Models.TransformerEncoder import PositionEncoding, PatchEmbedding, EncoderBlock


class MixModelTransformer(LightningModule):
    def __init__(self, module_dict, img_sizes, patch_size, embed_dim, in_channels, depth=3, wf=5, num_layers=12, loss_fcn=torch.nn.BCEWithLogitsLoss()):
        super().__init__()
        self.module_dict = module_dict
        self.pe = nn.ModuleList(
            [PositionEncoding(img_size=img_sizes[i], patch_size=patch_size, in_channel=2**(wf+i), embed_dim=embed_dim, dropout=0.8) for i in range(depth)]
        )
        self.layers = nn.ModuleList(
            [EncoderBlock(num_heads=8, embed_dim=embed_dim, mlp_dim=128, dropout=0.0) for _ in range(num_layers)])

        self.classifier = nn.Sequential(
            nn.LazyLinear(128),
            nn.LazyLinear(1)
        )
        self.accuracy = torchmetrics.AUC(reorder=True)
        self.loss_fcn = torch.nn.MSELoss()  # loss_fcn

    def forward(self, datadict):
        # features = torch.cat([self.module_dict[key](datadict[key]) for key in self.module_dict.keys()], dim=1)
        # For transformer
        flg = 0
        for key in self.module_dict.keys():
            if "Dose" == key or "Anatomy" == key:
                x = datadict[key]
                for i, down in enumerate(self.module_dict[key].model.encoder):
                    x = down(x)
                    if flg == 0:
                        feature = self.pe[i](x)
                        flg = 1
                    else:
                        out_trans = self.pe[i](x)
                        feature = torch.cat((feature, out_trans), dim=1)

                for layer in self.layers:
                    x = layer(feature)
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
        print('val_prediction:', prediction, label)
        print('val_loss:', val_loss)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return [optimizer], [scheduler]

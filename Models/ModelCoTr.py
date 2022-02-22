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

# Please refer to model CoTr: Efficiently Bridging CNN and Transformer for 3D Medical Image Segmentation.


class ModelCoTr(LightningModule):
    def __init__(self, module_dict, img_sizes, patch_size, embed_dim, in_channels, depth=3, wf=5, num_layers=3,
                 loss_fcn=torch.nn.BCEWithLogitsLoss()):
        super().__init__()
        self.module_dict = module_dict
        self.pe = PositionEncoding(img_size=[img_sizes[0]], patch_size=patch_size, in_channel=128,
                                   embed_dim=8 * 128, img_dim=3, dropout=0.5, iftoken=True)
        # self.pe = nn.ModuleList(
        #     [PositionEncoding(img_size=[img_sizes[i]], patch_size=patch_size, in_channel=2 ** (wf + i),
        #                       embed_dim=embed_dim, img_dim=3, dropout=0.5, iftoken=True) for i in range(depth)]
        # )
        self.transformers = nn.ModuleList(
            [TransformerBlock(num_heads=16, embed_dim=embed_dim, mlp_dim=128, dropout=0.5) for _ in range(num_layers)])

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

                feature = self.pe(x)
                # for i, down in enumerate(self.module_dict[key].model.encoder):
                #     x = down(x)
                #     if flg == 0:
                #         feature = self.pe[i](x)
                #         flg = 1
                #     else:
                #         out_trans = self.pe[i](x)
                #         feature = torch.cat((feature, out_trans), dim=1)

                for transformer in self.transformers:
                    x = transformer(feature)
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
        self.log("loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        datadict, label = batch
        prediction = self.forward(datadict)
        val_loss = self.loss_fcn(prediction.squeeze(dim=1), batch[-1])
        self.log("val_loss", val_loss, on_epoch=True)
        MAE = torch.abs(prediction.flatten(0) - label)
        out = {'MAE': MAE, 'img': datadict['Anatomy']}
        return out

    def generate_report(self, img):
        # img_batch = batch[0]['Anatomy']
        # tensorboard = self.logger.experiment
        img_batch = img.view(img.shape[0] * img.shape[1], *[1, img.shape[2], img.shape[3]])
        grid = torchvision.utils.make_grid(img_batch)
        return grid
        # tensorboard.add_image('worst_case_img', grid, self.current_epoch)

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
        prediction = self.forward(datadict)
        test_loss = self.loss_fcn(prediction.squeeze(dim=1), batch[-1])
        print('test_prediction:', prediction, label)
        print('test_loss:', test_loss)
        return test_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return [optimizer], [scheduler]

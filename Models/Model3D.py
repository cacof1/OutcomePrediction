import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from torchinfo import summary
from Models.unet3d import UNet3D
import torchmetrics
from monai.networks import blocks
from torch import nn


## Model
class Model3D(LightningModule):
    def __init__(self, config, wf=2, depth=3):
        super().__init__()
        self.config = config
        self.in_channels = config['MODEL']['in_channels']
        self.backbone = UnetEncoder(in_channels=self.in_channels, depth=depth, wf=wf)
        self.model = torch.nn.Sequential(
            self.backbone,
            torch.nn.Flatten(),
        )
        self.model.apply(self.weights_init)
        summary(self.model.to('cuda'), (3, 1, 20, 80, 80), col_names=["input_size", "output_size"], depth=5)
        self.accuracy = torchmetrics.AUC(reorder=True)
        self.loss_fcn = torch.nn.BCEWithLogitsLoss()

    def forward(self, x):
        # import pdb; pdb.set_trace()
        return self.model(x)

    def weights_init(self, m):
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)


class UnetEncoder(nn.Module):
    def __init__(self, depth, wf, in_channels):
        super(UnetEncoder, self).__init__()
        self.encoder = nn.ModuleList()
        for i in range(depth):
            out_channels = 2 ** (wf + i)
            down_block = blocks.UnetResBlock(spatial_dims=3, in_channels=in_channels, out_channels=out_channels,
                                             kernel_size=[5, 5, 5],
                                             stride=[2, 2, 2], norm_name='batch', dropout=0.5)
            self.encoder.append(down_block)
            in_channels = out_channels

        self.out_channels = in_channels

    def forward(self, x):
        for i, down in enumerate(self.encoder):
            x = down(x)
        return x

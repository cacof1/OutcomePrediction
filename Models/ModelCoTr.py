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
from Models.unet3d import UNet3D
## Models
from Models.Linear import Linear
from Models.Classifier2D import Classifier2D
from Models.Classifier3D import Classifier3D
from Models.TransformerEncoder import PositionEncoding, PatchEmbedding, TransformerBlock

# Please refer to model CoTr: Efficiently Bridging CNN and Transformer for 3D Medical Image Segmentation.


class ModelCoTr(LightningModule):
    def __init__(self, n_classes=1, wf=5, depth=3, img_sizes=256, patch_size=4, embed_dim=256,
                 in_channels=1, num_layers=3, num_heads=8, dropout=0.5, mlp_dim=128):
        super().__init__()
        self.model = UNet3D(in_channels=1, n_classes=n_classes, depth=depth, wf=wf).encoder
        self.pe = nn.ModuleList(
            [PositionEncoding(img_size=img_sizes[i], patch_size=patch_size, in_channel=2 ** (wf + i),
                              embed_dim=embed_dim, img_dim=3, dropout=dropout, iftoken=True) for i in range(depth)]
        )
        self.transformers = nn.ModuleList(
            [TransformerBlock(num_heads=num_heads, embed_dim=embed_dim, mlp_dim=mlp_dim, dropout=dropout) for _ in range(num_layers)])

    def forward(self, x):
        flg = 0
        for i, down in enumerate(self.model.encoder):
            x = down(x)
            if flg == 0:
                feature = self.pe[i](x)
                flg = 1
            else:
                out_trans = self.pe[i](x)
                feature = torch.cat((feature, out_trans), dim=1)

        for transformer in self.transformers:
            feature = transformer(feature)
        features = feature.flatten(start_dim=1)

        return features

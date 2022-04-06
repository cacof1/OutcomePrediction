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
from torchinfo import summary
## Models
from Models.Linear import Linear
from Models.Classifier2D import Classifier2D
from Models.Classifier3D import Classifier3D
from Models.TransformerEncoder import PositionEncoding, PatchEmbedding, TransformerBlock

# Please refer to model TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation


class ModelTransUnet(LightningModule):
    def __init__(self,  n_classes=1, wf=5, depth=3, img_sizes=256, patch_size=4, transformer_embed_dim=256, in_channels=1,
                 transformer_layer=3, transformer_head=8, dropout=0.5, transformer_mlp_dim=128):
        super().__init__()

        self.model = UNet3D(in_channels=1, n_classes=n_classes, depth=depth, wf=wf).encoder
        self.model.apply(self.weights_init)
        summary(self.model.to('cuda'), (3, 1, 32, 128, 128), col_names=["input_size", "output_size"], depth=5)

        self.pe = PositionEncoding(img_size=img_sizes, patch_size=patch_size, in_channel=in_channels,
                                   embed_dim=transformer_embed_dim, img_dim=3, dropout=dropout, iftoken=True)
        self.transformers = nn.ModuleList(
            [TransformerBlock(num_heads=transformer_head, embed_dim=transformer_embed_dim, mlp_dim=transformer_mlp_dim,
                              dropout=dropout) for _ in range(transformer_layer)])

    def forward(self, x):
        for i, down in enumerate(self.model.encoder):
            x = down(x)
        feature = self.pe(x)
        for transformer in self.transformers:
            feature = transformer(feature)
        features = feature.flatten(start_dim=1)

        return features

    def weights_init(self, m):
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)

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
from Models.UnetEncoder import UnetEncoder
from torchinfo import summary
## Models
from Models.Linear import Linear
from Models.Classifier2D import Classifier2D
from Models.Classifier3D import Classifier3D
from Models.TransformerEncoder import PositionEncoding, PatchEmbedding, TransformerBlock

# Please refer to model TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation
# img_sizes=256, patch_size=4, embed_dim=256, in_channels=1,
#                  num_layers=3, num_heads=8, dropout=0.5, mlp_dim=128

class ModelTransUnet(LightningModule):
    def __init__(self, config):
        super().__init__()
        parameters = config['MODEL_PARAMETERS']
        self.model = UnetEncoder(**parameters)
        self.model.apply(self.weights_init)
        summary(self.model.to('cuda'), (3, 1, 32, 128, 128), col_names=["input_size", "output_size"], depth=5)

        self.pe = PositionEncoding(img_size=config['img_sizes'], patch_size=config['patch_size'], in_channel=config['in_channels'],
                                   embed_dim=config['transformer_embed_dim'], img_dim=3, dropout=config['dropout'], iftoken=True)
        self.transformers = nn.ModuleList(
            [TransformerBlock(num_heads=config['transformer_head'], embed_dim=config['transformer_embed_dim'],
                              mlp_dim=config['transformer_mlp_dim'],
                              dropout=config['dropout']) for _ in range(config['transformer_layer'])])

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

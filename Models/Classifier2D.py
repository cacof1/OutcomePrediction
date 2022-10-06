import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchvision import models
from monai.networks import nets
from Models.UnetEncoder import UnetEncoder

class Classifier2D(pl.LightningModule):
    def __init__(self, config):

        super().__init__()
        self.config   = config
        self.backbone = getattr(models, config['MODEL']['Backbone'])
        if 'densenet' in config['MODEL']['Backbone']:
            self.backbone = self.backbone(pretrained=config['MODEL']['Pretrained'],drop_rate=config['MODEL']['Drop_Rate'])
        else: self.backbone = self.backbone(pretrained=config['MODEL']['Pretrained'])            

    def forward(self, x):
        return self.backbone(x)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)



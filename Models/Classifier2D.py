import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchvision import models
from monai.networks import nets
from Models.UnetEncoder import UnetEncoder


class Classifier2D(pl.LightningModule):
    def __init__(self, config, module_str):

        super().__init__()
        self.config = config
        # if 'densenet' in config['MODEL']['Backbone']:
        #     self.backbone = self.backbone(pretrained=config['MODEL']['Pretrained'],
        #                                   drop_rate=config['MODEL']['Drop_Rate'])
        # else:
        #     self.backbone = self.backbone(pretrained=config['MODEL']['Pretrained'])

        model = config['MODEL'][module_str + '_Backbone']
        parameters = config[module_str + '_MODEL_PARAMETERS']
        if model == 'Vit':
            self.backbone = models.vit_b_16(pretrained=True).eval()
        elif model == 'Unet':
            self.backbone = UnetEncoder(**parameters)
        else:
            model_str = 'nets.' + model + '(**parameters)'
            self.backbone = eval(model_str)

        layers = list(self.backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        return self.feature_extractor(x)

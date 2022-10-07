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
        model = config['MODEL'][module_str + '_Backbone']
        parameters = config[module_str + '_MODEL_PARAMETERS']
        if model == 'torchvision':
            model_name = config['MODEL'][module_str + '_model_name']
            model_str = 'models.' + model_name + '(pretrained=True)'
            self.backbone = eval(model_str)
        else:
            model_str = 'nets.' + model + '(**parameters)'
            self.backbone = eval(model_str)

        layers = list(self.backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

    def forward(self, x):
        features = self.feature_extractor(x)
        return features

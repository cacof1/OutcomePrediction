import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from pytorch_lightning import LightningModule
from torch import nn
import torchmetrics
from monai.networks import blocks, nets
from Models.UnetEncoder import UnetEncoder
from Models.PretrainedEncoder3D import PretrainedEncoder3D


## Model
class Classifier(LightningModule):
    def __init__(self, config, module_str):
        super().__init__()

        model = config['MODEL']['backbone']
        parameters = config['MODEL_PARAMETERS']

        # only use network for features
        if model == 'torchvision':
            model_name = config['MODEL'][module_str + '_model_name']
            model_str = 'models.' + model_name + '(pretrained=True)'
            self.backbone = eval(model_str)
        else:
            model_str = 'nets.' + model + '(**parameters)'
            self.backbone = eval(model_str)

        layers = list(self.backbone.children())[:-1] ## N->embedding
        self.model = nn.Sequential(*layers)

        self.flatten = nn.Sequential(
            # nn.Dropout(0.3),
            # nn.AdaptiveAvgPool3d(output_size=(4, 4, 4)),
            nn.Dropout(config['MODEL']['dropout_prob']),
            nn.AdaptiveAvgPool3d(output_size=(1, 1, 1)),
            nn.Flatten(),
        )
        self.model.apply(self.weights_init)

    def forward(self, x):
        features = self.model(x)
        return self.flatten(features)

    def weights_init(self, m):
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)





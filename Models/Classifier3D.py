import matplotlib.pyplot as plt
import torch
from pytorch_lightning import LightningModule
from torch import nn
import torchmetrics
from monai.networks import blocks, nets
from Models.UnetEncoder import UnetEncoder
from Models.PretrainedEncoder3D import PretrainedEncoder3D
## Model
class Classifier3D(LightningModule):
    def __init__(self, config, module_str):
        super().__init__()

        backbone_name = config['MODEL'][module_str + '_Backbone']
        parameters = config[module_str + '_MODEL_PARAMETERS']

        if backbone_name == 'Unet':
            self.backbone = UnetEncoder(**parameters)
        elif backbone_name == 'UNETR':
            self.backbone = PretrainedEncoder3D(config, module_str)
        else:
            model_str = 'nets.' + backbone_name + '(**parameters)'
            loaded_model = eval(model_str)
            # only use network for features
            self.backbone = loaded_model.features

        self.model = torch.nn.Sequential(
            self.backbone,
            torch.nn.Flatten(),
            torch.nn.AdaptiveAvgPool1d(128),
        )
        self.model.apply(self.weights_init)
        self.accuracy = torchmetrics.AUC(reorder=True)
        self.loss_fcn = torch.nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def weights_init(self, m):
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)




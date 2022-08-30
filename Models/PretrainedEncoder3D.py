import matplotlib.pyplot as plt
import torch
from pytorch_lightning import LightningModule
from torch import nn
import torchmetrics
from monai.networks import blocks, nets
from Models.UnetEncoder import UnetEncoder


class PretrainedEncoder3D(LightningModule):
    def __init__(self, config, module_str):
        super().__init__()
        self.n_classes = 1
        model_str = config['MODEL'][module_str + '_Backbone']
        parameters = config[module_str + '_MODEL_PARAMETERS']
        self.config = config

        self.loss_fcn = getattr(torch.nn, self.config["MODEL"]["Loss_Function"])()
        self.activation = getattr(torch.nn, self.config["MODEL"]["Activation"])()

        model_str = 'nets.' + model_str + '(**parameters)'
        full_model = eval(model_str)
        vit_dict = torch.load(config['MODEL'][module_str + '_ckpt_path'])
        vit_weights = vit_dict['state_dict']
        model_dict = full_model.state_dict()
        vit_weights = {k: v for k, v in vit_weights.items() if k in model_dict}
        model_dict.update(vit_weights)
        full_model.load_state_dict(model_dict)
        del model_dict, vit_weights, vit_dict

        self.backbone = full_model.vit
        self.hidden_size = full_model.hidden_size
        self.feat_size = full_model.feat_size
        self.encoder2 = full_model.encoder2
        self.encoder1 = full_model.encoder1

        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.encoder1.parameters():
            param.requires_grad = False

        for param in self.encoder2.parameters():
            param.requires_grad = False

        self.accuracy = torchmetrics.AUC(reorder=True)

    def forward(self, img):
        out1 = self.backbone(img)
        enc1 = self.encoder1(img)
        f1 = nn.AdaptiveMaxPool3d((1, 1, 48))(enc1)
        f1f = f1.flatten(start_dim=1)
        enc2 = self.encoder2(self.proj_feat(out1[1][3], self.hidden_size, self.feat_size))
        f2 = nn.AdaptiveAvgPool3d((1, 1, 12))(enc2)
        f2f = f2.flatten(start_dim=1)
        connect_features = torch.cat((f1f, f2f), dim=1)
        return connect_features

import torch
from torch import nn
from pytorch_lightning import LightningModule
from Models.TransformerEncoder import PositionEncoding, TransformerBlock
from torchvision import models
from monai.networks import nets

# Please refer paper CAE-TRANSFORMER: TRANSFORMER-BASED MODEL TO PREDICT INVASIVENESS
# OF LUNG ADENOCARCINOMA SUBSOLID NODULES FROM NON-THIN SECTION 3D
# CT SCANS


class Model2DNet(LightningModule):
    def __init__(self, config):
        super().__init__()
        ## define backbone
        self.config = config
        model = config['MODEL']['Backbone']

        self.linear1 = nn.LazyLinear(72)
        if model == 'Vit':
            self.backbone = models.vit_b_16(pretrained=True).eval()
        else:
            parameters = config['MODEL_PARAMETERS']
            model_str = 'nets.' + model + '(**parameters)'
            loaded_model = eval(model_str)
            self.backbone = loaded_model.features

        # self.backbone.eval()
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

        layers = list(self.backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        self.feature_extractor.eval()
        for param in self.feature_extractor[0:int(len(self.feature_extractor))].parameters():
            param.requires_grad = False

    def convert2d(self, x):
        if self.config['MODEL_PARAMETERS']['in_channels'] == 3:
            x = x.repeat(1, 3, 1, 1)
        features = self.backbone(x)
        features = features.flatten(1)
        features = self.linear1(features)
        features = features.unsqueeze(0)
        features = features.unsqueeze(1)
        # features = features.permute(2, 3, 0, 1)
        return features

    def forward(self, x):
        features = torch.cat([self.convert2d(b.transpose(0, 1)) for i, b in enumerate(x)], dim=0)
        features = features.flatten(start_dim=1)
        return features

import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from pytorch_lightning import LightningModule
from torch import nn
from torch._dynamo import OptimizedModule
import torchmetrics
from monai.networks import blocks, nets
from Models.UnetEncoder import UnetEncoder
from Models.PretrainedEncoder3D import PretrainedEncoder3D
import os
from totalsegmentator.python_api import totalsegmentator
from copy import deepcopy


## Model
class Classifier(LightningModule):
    def __init__(self, config, module_str):
        super().__init__()

        self.backbone_fixed = config['MODEL']['backbone_fixed']
        model = config['MODEL']['backbone']
        parameters = config['MODEL_PARAMETERS']

        # only use network for features
        if model == 'torchvision':
            model_name = config['MODEL'][module_str + '_model_name']
            model_str = 'models.' + model_name + '(pretrained=True)'
            self.backbone = eval(model_str)
            layers = list(self.backbone.children())[:-1]  ## N->embedding
        elif model == 'totalsegmentator':
            # totalsegmentator(config['MODEL']['backbone_folder'], config['MODEL']['backbone_folder'], fast=True)
            os.environ["nnUNet_raw"] = str(config['MODEL']['backbone_folder'])  # not needed, just needs to be an existing directory
            os.environ["nnUNet_preprocessed"] = str(config['MODEL']['backbone_folder'])  # not needed, just needs to be an existing directory
            os.environ["nnUNet_results"] = str(config['MODEL']['backbone_folder'])
            from nnunetv2.utilities.file_path_utilities import get_output_folder
            from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
            resample = 3.0
            task_id = 297
            trainer = 'nnUNetTrainer_4000epochs_NoMirroring'
            model = '3d_fullres'
            plans = 'nnUNetPlans'
            folds = [0]
            chk = "checkpoint_final.pth"
            step_size = 0.5
            unet_predictor = nnUNetPredictor(tile_step_size=step_size)
            model_folder = get_output_folder(task_id, trainer, plans, model)
            unet_predictor.initialize_from_trained_model_folder(
                model_folder,
                use_folds=folds,
                checkpoint_name=chk,
            )
            self.backbone = unet_predictor.network
            self.backbone.load_state_dict(unet_predictor.list_of_parameters[0])  # only one element
            encoder = self.backbone.encoder
            encoder = self.add_channels_to_ts(encoder, config['DATA']['n_channel'])
            layers = list(encoder.children())
        else:
            model_str = 'nets.' + model + '(**parameters)'
            self.backbone = eval(model_str)
            layers = list(self.backbone.children())[:-1] ## N->embedding

        self.model = nn.Sequential(*layers)

        if self.backbone_fixed:
            self.model.requires_grad_(False)
            self.model.train(False)

        # self.flatten = nn.Sequential(
        #     # nn.Dropout(0.3),
        #     # nn.AdaptiveAvgPool3d(output_size=(4, 4, 4)),
        #     nn.Dropout(config['MODEL']['dropout_prob']),
        #     nn.AdaptiveAvgPool3d(output_size=(1, 1, 1)),
        #     nn.Flatten(),
        # )

        self.flatten = nn.Sequential(
            nn.Dropout(config['MODEL']['dropout_prob']),
            nn.AdaptiveAvgPool3d(output_size=config['MODEL']['bottleneck']),
            nn.Flatten(),
            nn.Dropout(config['MODEL']['dropout_prob']),
            nn.Linear(
                config['MODEL']['backbone_out_c']*config['MODEL']['bottleneck'][0]*config['MODEL']['bottleneck'][1]
                * config['MODEL']['bottleneck'][2], config['MODEL']['classifier_in']),
        )

        if not config['MODEL']['pretrained']:
            self.model.apply(self.weights_init)

        self.flatten.apply(self.weights_init)

    def forward(self, x):
        return self.flatten(self.model(x))

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)

    @staticmethod
    def add_channels_to_ts(model, in_channels):
        old_first_layer = model.stages[0][0].convs[0].conv
        weights = old_first_layer.weight
        new_first_layer = nn.Conv3d(
            in_channels, old_first_layer.out_channels, old_first_layer.kernel_size, old_first_layer.stride,
            old_first_layer.padding, old_first_layer.dilation, old_first_layer.groups, bias=True,
            padding_mode=old_first_layer.padding_mode, device=old_first_layer.weight.device)
        # First dimension of filters same as pretrained
        new_weights = new_first_layer.weight.detach()
        new_weights[:, [0], :] = weights
        # All other dimensions of filters set to zero -> output of model with 3 channels will be the same as with 1
        for c in range(1, in_channels):
            new_weights[:, c, :] = 0
        new_first_layer.weight = nn.Parameter(new_weights)
        new_first_layer.bias = old_first_layer.bias
        model.stages[0][0].convs[0].conv = new_first_layer
        model.stages[0][0].convs[0].all_modules[0] = new_first_layer
        model.stages[0][0].convs[0].input_channels = 3
        return model


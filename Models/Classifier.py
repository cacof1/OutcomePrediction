import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from pytorch_lightning import LightningModule
from torch import nn
from torch._dynamo import OptimizedModule
import torchmetrics
from monai.networks import blocks, nets
#from .UnetEncoder import UnetEncoder
#from .PretrainedEncoder3D import PretrainedEncoder3D
from .SimpleCNNTesting import SimpleCNN
import os
from totalsegmentator.python_api import totalsegmentator
from copy import deepcopy

from torch.utils import model_zoo
from monai.utils import look_up_option
import re


## Model
class Classifier(LightningModule):
    def __init__(self, config, module_str):
        super().__init__()
        self.config = config
        self.backbone_fixed = config['MODEL']['backbone_fixed']
        model = config['MODEL']['backbone']
        parameters = config['MODEL_PARAMETERS']

        # only use network for features
        if model == 'torchvision':
            model_name = config['MODEL'][module_str + '_model_name']
            model_str = 'models.' + model_name + '(pretrained=True)'
            self.backbone = eval(model_str)
            layers = list(self.backbone.children())[:-1]  ## N->embedding
            self.model = nn.Sequential(*layers)
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
            self.model = nn.Sequential(*layers)
        elif model == 'simpleCNN':
            self.backbone = SimpleCNN(config, use_residual=False, use_dropout=False)
            self.model = self.backbone
        elif model == 'efficientnet':
            self.backbone = nets.EfficientNetBN("efficientnet-b0",
                                                spatial_dims=3,
                                                in_channels=self.config['DATA']['n_channel'],
                                                num_classes=self.config['DATA']['n_classes'],
                                                pretrained=self.config['MODEL']['pretrained'],  # added for testing
                                                )
            # self.backbone = EfficientNetBN("efficientnet-b0",
            #                                spatial_dims=3,
            #                                in_channels=self.config['DATA']['n_channel'],
            #                                num_classes=self.config['DATA']['n_classes'],
            #                                pretrained=self.config['MODEL']['pretrained'],  # added for testing
            #                                )
            self.model = self.backbone
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
        return (self.flatten(self.model(x))
                if self.config['MODEL']['backbone'] not in ['simpleCNN', 'efficientnet'] else self.model(x))

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


class EfficientNetBN(nets.EfficientNet):

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        progress: bool = True,
        spatial_dims: int = 2,
        in_channels: int = 3,
        num_classes: int = 1000,
        norm = ("batch", {"eps": 1e-3, "momentum": 0.01}),
        adv_prop: bool = False,
    ) -> None:
        """
        Generic wrapper around EfficientNet, used to initialize EfficientNet-B0 to EfficientNet-B7 models
        model_name is mandatory argument as there is no EfficientNetBN itself,
        it needs the N in [0, 1, 2, 3, 4, 5, 6, 7, 8] to be a model

        Args:
            model_name: name of model to initialize, can be from [efficientnet-b0, ..., efficientnet-b8, efficientnet-l2].
            pretrained: whether to initialize pretrained ImageNet weights, only available for spatial_dims=2 and batch
                norm is used.
            progress: whether to show download progress for pretrained weights download.
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            num_classes: number of output classes.
            norm: feature normalization type and arguments.
            adv_prop: whether to use weights trained with adversarial examples.
                This argument only works when `pretrained` is `True`.

        Examples::

            # for pretrained spatial 2D ImageNet
            >>> image_size = get_efficientnet_image_size("efficientnet-b0")
            >>> inputs = torch.rand(1, 3, image_size, image_size)
            >>> model = EfficientNetBN("efficientnet-b0", pretrained=True)
            >>> model.eval()
            >>> outputs = model(inputs)

            # create spatial 2D
            >>> model = EfficientNetBN("efficientnet-b0", spatial_dims=2)

            # create spatial 3D
            >>> model = EfficientNetBN("efficientnet-b0", spatial_dims=3)

            # create EfficientNetB7 for spatial 2D
            >>> model = EfficientNetBN("efficientnet-b7", spatial_dims=2)

        """
        # block args
        blocks_args_str = [
            "r1_k3_s11_e1_i32_o16_se0.25",
            "r2_k3_s22_e6_i16_o24_se0.25",
            "r2_k5_s22_e6_i24_o40_se0.25",
            "r3_k3_s22_e6_i40_o80_se0.25",
            "r3_k5_s11_e6_i80_o112_se0.25",
            "r4_k5_s22_e6_i112_o192_se0.25",
            "r1_k3_s11_e6_i192_o320_se0.25",
        ]

        # check if model_name is valid model
        if model_name not in efficientnet_params:
            model_name_string = ", ".join(efficientnet_params.keys())
            raise ValueError(f"invalid model_name {model_name} found, must be one of {model_name_string} ")

        # get network parameters
        weight_coeff, depth_coeff, image_size, dropout_rate, dropconnect_rate = efficientnet_params[model_name]

        # create model and initialize random weights
        super().__init__(
            blocks_args_str=blocks_args_str,
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            num_classes=num_classes,
            width_coefficient=weight_coeff,
            depth_coefficient=depth_coeff,
            dropout_rate=dropout_rate,
            image_size=image_size,
            drop_connect_rate=dropconnect_rate,
            norm=norm,
        )

        # only pretrained for when `spatial_dims` is 2
        if pretrained and (spatial_dims == 2):
            _load_state_dict(self, model_name, progress, adv_prop)


efficientnet_params = {
    # model_name: (width_mult, depth_mult, image_size, dropout_rate, dropconnect_rate)
    "efficientnet-b0": (1.0, 1.0, 224, 0.2, 0.2),
    "efficientnet-b1": (1.0, 1.1, 240, 0.2, 0.2),
    "efficientnet-b2": (1.1, 1.2, 260, 0.3, 0.2),
    "efficientnet-b3": (1.2, 1.4, 300, 0.3, 0.2),
    "efficientnet-b4": (1.4, 1.8, 380, 0.4, 0.2),
    "efficientnet-b5": (1.6, 2.2, 456, 0.4, 0.2),
    "efficientnet-b6": (1.8, 2.6, 528, 0.5, 0.2),
    "efficientnet-b7": (2.0, 3.1, 600, 0.5, 0.2),
    "efficientnet-b8": (2.2, 3.6, 672, 0.5, 0.2),
    "efficientnet-l2": (4.3, 5.3, 800, 0.5, 0.2),
}


url_map = {
    "efficientnet-b0": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth",
    "efficientnet-b1": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth",
    "efficientnet-b2": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth",
    "efficientnet-b3": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth",
    "efficientnet-b4": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth",
    "efficientnet-b5": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth",
    "efficientnet-b6": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b6-c76e70fd.pth",
    "efficientnet-b7": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth",
    # trained with adversarial examples, simplify the name to decrease string length
    "b0-ap": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b0-b64d5a18.pth",
    "b1-ap": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b1-0f3ce85a.pth",
    "b2-ap": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b2-6e9d97e5.pth",
    "b3-ap": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b3-cdd7c0f4.pth",
    "b4-ap": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b4-44fb3a87.pth",
    "b5-ap": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b5-86493f6b.pth",
    "b6-ap": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b6-ac80338e.pth",
    "b7-ap": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b7-4652b6dd.pth",
    "b8-ap": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/adv-efficientnet-b8-22a8fe65.pth",
}


def _load_state_dict(model: nn.Module, arch: str, progress: bool, adv_prop: bool) -> None:
    if adv_prop:
        arch = arch.split("efficientnet-")[-1] + "-ap"
    model_url = look_up_option(arch, url_map, None)
    if model_url is None:
        print(f"pretrained weights of {arch} is not provided")
    else:
        # load state dict from url
        model_url = url_map[arch]
        pretrain_state_dict = model_zoo.load_url(model_url, progress=progress)
        model_state_dict = model.state_dict()

        pattern = re.compile(r"(.+)\.\d+(\.\d+\..+)")
        for key, value in model_state_dict.items():
            pretrain_key = re.sub(pattern, r"\1\2", key)
            if pretrain_key in pretrain_state_dict and value.shape == pretrain_state_dict[pretrain_key].shape:
                model_state_dict[key] = pretrain_state_dict[pretrain_key]

        model.load_state_dict(model_state_dict)


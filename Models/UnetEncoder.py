from torch import nn
from monai.networks import blocks, nets

class UnetEncoder(nn.Module):
    def __init__(self, depth, wf, in_channels, spatial_dims=3, kernel_size=None, stride=None):
        super(UnetEncoder, self).__init__()
        self.encoder = nn.ModuleList()
        for i in range(depth):
            out_channels = 2 ** (wf + i)
            down_block = blocks.UnetResBlock(spatial_dims=spatial_dims, in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=kernel_size,
                                             stride=stride, norm_name='batch', dropout=0.5)
            self.encoder.append(down_block)
            in_channels = out_channels

        self.out_channels = in_channels

    def forward(self, x):
        for i, down in enumerate(self.encoder):
            x = down(x)
        return x


    def weights_init(self, m):
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
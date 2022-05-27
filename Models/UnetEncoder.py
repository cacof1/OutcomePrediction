from torch import nn

class UnetEncoder(nn.Module):
    def __init__(self, depth, wf, in_channels):
        super(UnetEncoder, self).__init__()
        self.encoder = nn.ModuleList()
        for i in range(depth):
            out_channels = 2 ** (wf + i)
            down_block = blocks.UnetResBlock(spatial_dims=3, in_channels=in_channels, out_channels=out_channels,
                                             kernel_size=[5, 5, 5],
                                             stride=[2, 2, 2], norm_name='batch', dropout=0.5)
            self.encoder.append(down_block)
            in_channels = out_channels

        self.out_channels = in_channels

    def forward(self, x):
        for i, down in enumerate(self.encoder):
            x = down(x)
        return x
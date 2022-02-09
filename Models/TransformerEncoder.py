import torch
import torch.nn as nn


class PositionEncoding(nn.Module):

    def __init__(self, img_size, patch_size, in_channel,embed_dim,dropout=0.8):
        super().__init__()
        self.img_size = img_size
        self.in_channel = in_channel
        self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, in_channel=in_channel)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, C, H, W, D = x.shape
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)


        return x

    def get_attention_maps(self, x):
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, return_attention=True)
            attention_maps.append(attn_map)
            x = l(x)
        return attention_maps


class TransformerBlock(nn.Module):

    def __init__(self, num_heads, embed_dim, mlp_dim, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = self.query = nn.Linear(embed_dim, embed_dim)

        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)

        # Two-layer MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

        # Layers to apply in between the main layers
        self.norm0 = nn.LayerNorm(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        query = self.query(self.norm0(x))
        key = self.key(self.norm0(x))
        value = self.value(self.norm0(x))
        out, attention = self.attn(query, key, value)
        x = x + self.dropout(out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.mlp(x)
        x = x + linear_out
        x = self.norm2(x)

        return x

##
#class TransformerRegression(nn.Module):
#    def __init__(self, num_layers, input_dim, num_heads, embed_dim, mlp_dim, dropout=0.0,
#                 input_dropout=0.0):
#        super().__init__()
#        self.linear_net = nn.Sequential(
#            nn.Dropout(input_dim),
#            nn.Linear(input_dim, embed_dim)
#        )
#        self.transformer = TransformerEncoder(num_layers, num_heads, embed_dim, mlp_dim, dropout=0.0)
#
#    def forward(self, x):
#        linear_out = self.linear_net(x)
#        return self.transformer(linear_out)


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=64, patch_size=4, embed_dim = 64, in_channel=1):
        super().__init__()
        if len(img_size) == 1:
            num_patches = (img_size[0] // patch_size) ** 3 ## for 3D image
        if len(img_size) == 3:
            num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size) * (img_size[2] // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(in_channel, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W, D = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

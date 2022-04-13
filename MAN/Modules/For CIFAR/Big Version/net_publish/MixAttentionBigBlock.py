import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import torchsummary
from net_publish.cnn_attention.spatial_channel_attention import ConvAttentionBlock


device = "cuda:0" if torch.cuda.is_available() else "cpu"

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class PatchEmbed(nn.Module):
    def __init__(self, input_shape, in_chans, patch_size=2,  num_features=128, norm_layer=None,
                 flatten=True):
        super().__init__()
        self.num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, num_features, kernel_size=2, stride=2)
        self.norm = norm_layer(num_features) if norm_layer else nn.Identity()

    def forward(self, x):
        # B, 64, 32, 32 -> B, 64, 16, 16
        # print("x patch,", x.shape)
        x = self.proj(x)

        # B, 64, 16, 16 -> B, 64, 256 -> B, 256, 64
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


# -------------------------------------------------#
#   Basic Convolution Block
#   Conv2d + BatchNorm2d + LeakyReLU
# -------------------------------------------------#
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


# --------------------------------------------------------------------------------------------------------------------#
#   Global Attention
#   divide the feature into q, k, v
# --------------------------------------------------------------------------------------------------------------------#
class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = (drop, drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class ViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# darknet-style residual block
class Resblock_body(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Resblock_body, self).__init__()
        self.out_channels = out_channels

        self.conv1 = BasicConv(in_channels, out_channels, 3)

        self.conv2 = BasicConv(out_channels // 2, out_channels // 2, 3)
        self.conv3 = BasicConv(out_channels // 2, out_channels // 2, 3)

        self.conv4 = BasicConv(out_channels, out_channels, 1)
        self.maxpool = nn.MaxPool2d([2, 2], [2, 2])

    def forward(self, x):
        # 3*3 convolution
        x = self.conv1(x)
        # a long residual route from backbone
        route = x

        c = self.out_channels
        # split the channel and use the second part
        x = torch.split(x, c // 2, dim=1)[1]
        # 3*3 convolution on the backbone
        x = self.conv2(x)
        # a small residual route
        route1 = x
        # 3*3 convolution on the backbone
        x = self.conv3(x)
        # concatenate the backbone with residual block
        x = torch.cat([x, route1], dim=1)

        # 1*1 convolution
        x = self.conv4(x)

        x = torch.cat([route, x], dim=1)

        # maxpool downsampling
        x = self.maxpool(x)
        return x


class ViTAttentionBlock(nn.Module):
    def __init__(
            self, input_shape, num_features, patch_size=2, in_channels=3,
            depth=1, num_heads=4, mlp_ratio=4., qkv_bias=True, drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.2,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=GELU
    ):
        super(ViTAttentionBlock, self).__init__()
        # ViT
        self.patch_embed = PatchEmbed(input_shape=input_shape, patch_size=patch_size, in_chans=in_channels,
                                      num_features=num_features)
        num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
        self.num_features = num_features
        self.feature_shape = [int(input_shape[0] // patch_size), int(input_shape[1] // patch_size)]

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(
            *[
                ViTBlock(
                    dim=num_features,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer
                ) for i in range(depth)
            ]
        )

        # self.pos_embed = nn.Parameter(torch.zeros(in_channels, num_patches, num_features))
        self.pos_drop = nn.Dropout(p=0.1)
        self.norm = norm_layer(num_features)

    def forward_features(self, x):

        # batch, 3, 32, 32 -> batch, 64, 16, 16 -> batch, 256, 64
        x = self.patch_embed(x)

        # batch, 256, 64 -> 1, 16, 16, 64 -> 1, 64, 16, 16
        img_token_x = x.view(x.shape[0], *self.feature_shape, -1).permute(0, 3, 1, 2)

        # 1, 64, 16, 16 -> 1, 16, 16, 64 -> 1, 256, 64
        img_token_x = img_token_x.permute(0, 2, 3, 1).flatten(1, 2)

        # batch, 256, 64 + 1, 256, 64 -> batch, 256, 64
        # print("vit embed x:", x.shape)
        # print("vit embed img_token_x:", img_token_x.shape)
        x = self.pos_drop(x + img_token_x)

        # batch, 256, 64 -> batch, 256, 64
        x = self.blocks(x)
        x = self.norm(x)
        x = x.reshape(x.shape[0], *self.feature_shape, x.shape[-1]).permute(0, 3, 1, 2)
        return x

    def forward(self, x):
        # ViT
        x = self.forward_features(x)
        return x


class MixAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, input_shape):
        super().__init__()
        # compound residual block lets out_channels*2
        self.compound_res_block = Resblock_body(in_channels=in_channels, out_channels=out_channels)
        self.conv_attention = ConvAttentionBlock(channel=out_channels*2)

        # # 降通道卷积，使ConvAttention的通道和ViTAttention对上
        self.conv_down = nn.Conv2d(in_channels=out_channels*2, out_channels=out_channels, kernel_size=1, stride=1)

        self.vit_attention = ViTAttentionBlock(in_channels=in_channels, input_shape=input_shape, patch_size=2,
                                               num_features=out_channels*2)

    def forward(self, x):
        vit_x = x
        x = self.compound_res_block(x)

        # residual side
        res_x = x

        x = self.conv_attention(x)

        x = res_x + x

        x = x + self.vit_attention(vit_x)

        x = self.conv_down(x)

        return x


mix = MixAttentionBlock(in_channels=3, out_channels=256, input_shape=[224, 224]).to(device)
print(torchsummary.summary(mix, input_size=(3, 224, 224)))

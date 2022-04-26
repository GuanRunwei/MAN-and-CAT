import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import torchsummary


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

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
        print("x flatten shape:", x.shape)
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
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class ConvAttentionBlock(nn.Module):
    def __init__(self, channel, ratio=4, kernel_size=3):
        super(ConvAttentionBlock, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)

        return x


# --------------------------------------------------------------------------------------------------------------------#
#   Attention机制
#   将输入的特征qkv特征进行划分，首先生成query, key, value。query是查询向量、key是键向量、v是值向量。
#   然后利用 查询向量query 点乘 转置后的键向量key，这一步可以通俗的理解为，利用查询向量去查询序列的特征，获得序列每个部分的重要程度score。
#   然后利用 score 点乘 value，这一步可以通俗的理解为，将序列每个部分的重要程度重新施加到序列的值上去。
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
        # 利用一个3x3卷积进行特征整合
        x = self.conv1(x)
        # 引出一个大的残差边route
        route = x

        c = self.out_channels
        # 对特征层的通道进行分割，取第二部分作为主干部分。
        x = torch.split(x, c // 2, dim=1)[1]
        # 对主干部分进行3x3卷积
        x = self.conv2(x)
        # 引出一个小的残差边route_1
        route1 = x
        # 对第主干部分进行3x3卷积
        x = self.conv3(x)
        # 主干部分与残差部分进行相接
        x = torch.cat([x, route1], dim=1)

        # 对相接后的结果进行1x1卷积
        x = self.conv4(x)

        x = torch.cat([route, x], dim=1)

        # 利用最大池化进行高和宽的压缩
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
        # batch, 64, 32, 32 -> batch, 64, 1024 -> batch, 1024, 64
        x = self.patch_embed(x).contiguous()
        print("vit embed x:", x.shape)


        # batch, 256, 64 -> 1, 16, 16, 64 -> 1, 64, 16, 16
        print("feature shape:", self.feature_shape)
        img_token_x = x.view(x.shape[0], *self.feature_shape, -1).permute(0, 3, 1, 2)
        print("vit embed img_token_x1:", img_token_x.shape)

        # 1, 64, 16, 16 -> 1, 16, 16, 64 -> 1, 256, 64
        img_token_x = img_token_x.permute(0, 2, 3, 1).flatten(1, 2)

        # batch, 256, 64 + 1, 256, 64 -> batch, 256, 64

        print("vit embed img_token_x2:", img_token_x.shape)
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
        # compound会让通道 out_channels*2
        self.compound_res_block = Resblock_body(in_channels=in_channels, out_channels=out_channels)
        self.conv_attention = ConvAttentionBlock(channel=out_channels*2)

        # # 降通道卷积，使ConvAttention的通道和ViTAttention对上
        self.conv_down = nn.Conv2d(in_channels=out_channels*2, out_channels=out_channels, kernel_size=1, stride=1)

        self.vit_attention = ViTAttentionBlock(in_channels=in_channels, input_shape=input_shape, patch_size=2,
                                               num_features=out_channels*2)

    def forward(self, x):
        vit_x = x
        # print("初始x", x.shape)
        x = self.compound_res_block(x)
        # print("compound_x", x.shape)
        # residual side
        res_x = x

        x = self.conv_attention(x)
        # print("卷积注意力x", x.shape)
        x = res_x + x
        # print("conv attention:", x.shape)

        # print("vit attention:", self.vit_attention(x).shape)

        x = x + self.vit_attention(vit_x)

        x = self.conv_down(x)

        return x


class MANet(nn.Module):
    def __init__(self, input_shape, in_channels=3, num_class=100, pretrained=False):
        super(MANet, self).__init__()
        self.BasicConv1 = BasicConv(in_channels=in_channels, out_channels=64, kernel_size=1, stride=1)


        self.MixAttention1 = MixAttentionBlock(in_channels=64, out_channels=128, input_shape=input_shape)
        input_shape2 = [input_shape[0]/2, input_shape[1]/2]
        self.MixAttention2 = MixAttentionBlock(in_channels=128, out_channels=256, input_shape=input_shape2)
        input_shape3 = [input_shape2[0]/2, input_shape2[1]/2]
        self.MixAttention3 = MixAttentionBlock(in_channels=256, out_channels=512, input_shape=input_shape3)

        self.batchnorm1 = nn.BatchNorm2d(128)
        self.batchnorm2 = nn.BatchNorm2d(256)
        self.batchnorm3 = nn.BatchNorm2d(512)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_class)

    def forward(self, x):
        # x = F.interpolate(x, scale_factor=2, mode='bilinear')

        x = self.BasicConv1(x)

        print("img size:", x.shape)
        x = self.MixAttention1(x)

        x = self.batchnorm1(x)

        x = self.MixAttention2(x)
        # print(x.shape)
        x = self.batchnorm2(x)

        x = self.MixAttention3(x)
        x = self.batchnorm3(x)

        x = self.pool(x)
        x = x.reshape(x.shape[0], x.shape[-3]*x.shape[-2]*x.shape[-1])
        x = self.fc(x)

        return x




mix_attention_block = MixAttentionBlock(3, 32, [112, 112]).cuda()
print(torchsummary.summary(mix_attention_block, input_size=(3, 112, 112)))

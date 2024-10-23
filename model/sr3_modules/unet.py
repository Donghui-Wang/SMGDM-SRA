import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
import numpy as np
import pickle
import cv2
import os
from einops import rearrange, repeat


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


# 改进点1
# class FeatureWiseAffine(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         # 这些改进可以增强模型的学习能力，提高泛化性能，并有助于更有效地捕获和利用嵌入信息。
#         super(FeatureWiseAffine, self).__init__()
#         self.noise_func = nn.Sequential(
#             nn.Linear(in_channels, out_channels), # 将输入嵌入从 in_channels 映射到更大的尺寸 out_channels * 2，增加了模型的表示能力。
#             nn.BatchNorm1d(out_channels),  # 批次归一化，有助于稳定训练过程，可以加速收敛，并有助于防止过拟合。
#             nn.LeakyReLU(0.01),  # 非线性激活层LeakyReLU可能有助于模型捕获更复杂的特征变换关系。
#             nn.Dropout(0.05),  # 丢弃层 提供了一种形式的正则化，有助于防止过拟合。在训练过程中随机关闭一部分神经元，使模型不过分依赖于任何单一特征
#         )
#     def forward(self, x, noise_embed):
#         batch = x.shape[0]
#         x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
#         return x


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels * (1 + self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# building block modules

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


# 改进点3
class Block1(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            DepthwiseSeparableConv(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ChannelSqueezeExcitation(nn.Module):
    def __init__(self, channel, reduction_ratio=2):
        super(ChannelSqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channel, channel // reduction_ratio, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channel // reduction_ratio, channel, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialSqueezeExcitation(nn.Module):
    def __init__(self, channel):
        super(SpatialSqueezeExcitation, self).__init__()
        self.conv = nn.Conv2d(channel, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.sigmoid(self.conv(x))
        return x * y


class ChannelSpatialSqueezeExcitation(nn.Module):
    def __init__(self, channel, reduction_ratio=2):
        super(ChannelSpatialSqueezeExcitation, self).__init__()
        self.cse = ChannelSqueezeExcitation(channel, reduction_ratio)
        self.sse = SpatialSqueezeExcitation(channel)

    def forward(self, x):
        return self.cse(x) + self.sse(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()
        self.csse = ChannelSpatialSqueezeExcitation(dim_out)

    def forward(self, x, time_emb):
        b, c, h, w = x.shape
        h = self.block1(x)
        h = self.csse(h)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn == "SpatialSelfAttention":
            self.with_attn = SpatialSelfAttention(dim_out)
        elif with_attn == "SelfAttention":
            self.with_attn = SelfAttention(dim_out)
        elif with_attn == "MixedAttentionA":
            self.with_attn = MixedAttention(dim_out)
        else:
            self.with_attn = False

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if (self.with_attn):
            x = self.with_attn(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=4, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        # self.norm = nn.LayerNorm(in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)
        self.dropout = nn.Dropout(0.1)  # 添加 Dropout 正则化

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = self.dropout(attn)  # 应用 Dropout 正则化
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


# 改进点2
# 空间自注意力 残差连接
class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x + h_


class MixedAttention(nn.Module):
    def __init__(self, in_channels, n_head=4, norm_groups=32):
        super(MixedAttention, self).__init__()
        self.self_attention = SelfAttention(in_channels, n_head, norm_groups)
        self.spatial_attention = SpatialSelfAttention(in_channels)
        # 初始化权重为可学习的参数
        self.self_attn_weight = nn.Parameter(torch.tensor(0.7))

    def forward(self, x):
        x_self_attention = self.self_attention(x)
        x_spatial_attention = self.spatial_attention(x)
        # 使用可学习的权重
        return self.self_attn_weight * x_self_attention + (1 - self.self_attn_weight) * x_spatial_attention


class UNet(nn.Module):
    def __init__(
            self,
            in_channel=6,
            out_channel=3,
            inner_channel=32,
            norm_groups=32,
            channel_mults=(1, 2, 4, 8, 8),
            attn_res=(8),
            res_blocks=3,
            dropout=0,
            with_noise_level_emb=True,
            image_size=128
    ):
        super().__init__()

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]

        # downs = [InceptionSepConvBlock(in_channel, inner_channel,32,64,128)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                #             with_attn = "down"+str(ind)+str(_)
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                    dropout=dropout, with_attn=False))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res // 2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                               norm_groups=norm_groups,
                               dropout=dropout, with_attn="MixedAttentionA"),
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                               norm_groups=norm_groups,
                               dropout=dropout, with_attn="")
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks + 1):
                #               with_attn = "up"+str(ind) + str(_)
                ups.append(ResnetBlocWithAttn(
                    pre_channel + feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups,
                    dropout=dropout, with_attn=False))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res * 2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)

        self.mask_tail = FCN()

    def forward(self, x, time,continous):
        x_lr = x[:, :3, :, :]
        x_mask = x[:, 3, :, :].unsqueeze(1)
        x_noisy = x[:, 4:, :, :]
        # updated_mask = self.mask_update(x_noisy, x_mask)
        # x_updated_mask = updated_mask.detach()
        x = torch.cat((x_lr, x_mask, x_noisy), dim=1)

        t = self.noise_level_mlp(time) if exists(
            self.noise_level_mlp) else None

        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                feat = feats.pop()
                # if x.shape[2]!=feat.shape[2] or x.shape[3]!=feat.shape[3]:
                #     feat = F.interpolate(feat, x.shape[2:])
                x = layer(torch.cat((x, feat), dim=1), t)
            else:
                x = layer(x)

        return self.final_conv(x), self.mask_tail(x)





class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.dilated_conv1 = nn.Conv2d(64, 64, 3, padding=2, dilation=2)  # 空洞卷积
        self.dilated_conv2 = nn.Conv2d(64, 64, 3, padding=4, dilation=4)  # 空洞卷积
        self.final_conv = nn.Conv2d(64, 1, 1, 1, 0)
        self.sigmoid = nn.Sigmoid()

        # 调整 identity 通道数的卷积层
        self.match_channels = nn.Conv2d(64, 1, 1, 1, 0)

    def forward(self, x):
        identity = x  # 保存原始输入作为恒等（残差）连接

        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.dilated_conv1(x1))
        x3 = self.relu(self.dilated_conv2(x2))

        # 多尺度特征融合
        x_fused = x1 + x2 + x3

        # 应用最终卷积和激活函数
        out = self.sigmoid(self.final_conv(x_fused))

        # 将 identity 调整为与 out 相同的通道数
        identity = self.match_channels(identity)

        # 添加残差连接
        out = out + identity

        return out


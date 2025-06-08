import math

import torch

from einops import rearrange
from retention import MultiScaleRetention
from torch.nn import init

import torch.nn as nn
import numpy as np

NUM_CLASS = 16
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PConv2_1(nn.Module):
    def __init__(self, dim, kernel_size=3, n_div=2, ):
        super().__init__()
        self.dim_conv = dim // n_div
        self.conv = nn.Conv2d(self.dim_conv, self.dim_conv, kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_conv], dim=1)
        x1 = self.conv(x1)
        x = torch.concat([x1, x2], dim=1)

        return x

class PConv2_2(nn.Module):
    def __init__(self, dim, kernel_size=3, n_div=2, ):
        super().__init__()
        self.dim_conv = dim // n_div
        self.conv = nn.Conv2d(self.dim_conv, self.dim_conv, kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_conv], dim=1)
        x2 = self.conv(x2)
        x = torch.concat([x1, x2], dim=1)

        return x

class FasterNetBlock2_1(nn.Module):
    def __init__(self, dim, expand_ratio=2, act_layer=nn.ReLU, drop_path_rate=0.0):
        super().__init__()
        self.pconv1 = PConv2_1(dim)
        self.conv1 = nn.Conv2d(dim, int(dim * expand_ratio), 1, bias=False)
        self.bn = nn.BatchNorm2d(int(dim * expand_ratio))
        self.act_layer = act_layer()
        self.conv2 = nn.Conv2d(int(dim * expand_ratio), dim, 1, bias=False)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.eca_layer = Eca_layer()

    def forward(self, x):
        residual = x
        x = self.eca_layer(x)
        x = self.pconv1(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.act_layer(x)
        x = self.conv2(x)
        x = residual + self.drop_path(x)
        return x

class FasterNetBlock2_2(nn.Module):
    def __init__(self, dim=80, expand_ratio=2, act_layer=nn.ReLU, drop_path_rate=0.0):
        super().__init__()
        self.pconv2 = PConv2_2(dim)
        self.conv1 = nn.Conv2d(dim, int(dim * expand_ratio), 1, bias=False)
        self.bn = nn.BatchNorm2d(int(dim * expand_ratio))
        self.act_layer = act_layer()
        self.conv2 = nn.Conv2d(int(dim * expand_ratio), dim, 1, bias=False)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.eca_layer = Eca_layer()

    def forward(self, x):
        residual = x
        x = self.eca_layer(x)
        x = self.pconv2(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.act_layer(x)
        x = self.conv2(x)
        x = residual + self.drop_path(x)
        return x




class Eca_layer(nn.Module):
    """Constructs a ECA module.*
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self,):
        super(Eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        channel = x.shape[1]
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)  #y  64,40,1,1

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)
        values, indices = torch.topk(y, channel, dim=1, largest=True, sorted=True)
        b, c, h, w = x.shape
        # output = x * y.expand_as(x)
        # output = torch.sum(output, dim=1).unsqueeze(1)
        out = []
        for i in range(b):
            m = x[i, :, :, :]
            j = indices[i]
            j = torch.squeeze(j)
            t = m.index_select(0, j)
            t = torch.unsqueeze(t, 0)
            out.append(t)
        out = torch.cat(out, dim=0)
        # z = torch.cat([output, out], dim=1)
        return out




class Embeding(nn.Module):
    def __init__(self, channels=None, faster_dim=None, act_layer=nn.ReLU):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(channels, faster_dim, 3, stride=1, bias=False),
            nn.BatchNorm2d(faster_dim),
            act_layer()
        )

    def forward(self, x):
        x = self.stem(x)
        return x


class Mering(nn.Module):
    def __init__(self, faster_dim, faster_dim1, act_layer=nn.ReLU):
        super().__init__()
        self.up = nn.Sequential(
                        nn.Conv2d(faster_dim, faster_dim1, 3, stride=1, bias=False),
                        nn.BatchNorm2d(faster_dim1),
                        act_layer()
                    )

    def forward(self, x):
        x = self.up(x)
        return x



class FasterNet(nn.Module):
    def __init__(self,  channels,faster_dim,faster_dim1):#30,42,64
        super().__init__()

        self.embeding = Embeding(channels, faster_dim)
        self.mering1 = Mering(faster_dim, faster_dim1)
        self.FasterNetBlock2_1 =FasterNetBlock2_1(faster_dim)#
        self.FasterNetBlock2_2 =FasterNetBlock2_2(faster_dim1)
    def forward(self, x):  # x 64,30,13,13
        x = self.embeding(x)# 64，40，11，11
        #2层串行
        x = self.FasterNetBlock2_1(x)
        x = self.mering1(x)
        x = self.FasterNetBlock2_2(x)
        return x


class efr(nn.Module):
    def __init__(self, layers_ret1, ffn1,heads1,
                 hidden_dim=None, faster_dim=None,
                 dim2=None, num_classes=None, channels=None,
                 token_L=None, double_v_dim=False):
        super().__init__()
        # 添加
        print("...faster_dim{}".format(faster_dim))
        print("...hidden_dim1{}".format(hidden_dim))

        self.fasternet = FasterNet(channels, faster_dim, hidden_dim)#30,42,
        self.retnet1 = RetNet( layers_ret1, hidden_dim, ffn1*hidden_dim, heads1, double_v_dim=False)
        self.nn1 = nn.Linear((hidden_dim), num_classes)  #
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, X):
        """
        X: (batch_size, sequence_length, hidden_size)
        """
        X = torch.squeeze(X)  # 64,30,13,13
        X = self.fasternet(X)  #64,320,5,5
        X = rearrange(X, 'b c h w -> b (h w) c')
        X = self.retnet1(X)
        X = X.mean(dim=1)
        X = self.nn1(X)

        return X

    # def forward_recurrent(self, x_n, s_n_1s, n):
    #     """
    #     X: (batch_size, sequence_length, hidden_size)
    #     s_n_1s: list of lists of tensors of shape (batch_size, hidden_size // heads, hidden_size // heads)
    #
    #     """
    #     s_ns = []
    #     for i in range(self.layers):
    #         # list index out of range
    #         o_n, s_n = self.retentions[i].forward_recurrent(self.layer_norms_1[i](x_n), s_n_1s[i], n)
    #         y_n = o_n + x_n
    #         s_ns.append(s_n)
    #         x_n = self.ffns[i](self.layer_norms_2[i](y_n)) + y_n
    #
    #     return x_n, s_ns
    #
    # def forward_chunkwise(self, x_i, r_i_1s, i):
    #     """
    #     X: (batch_size, sequence_length, hidden_size)
    #     r_i_1s: list of lists of tensors of shape (batch_size, hidden_size // heads, hidden_size // heads)
    #
    #     """
    #     r_is = []
    #     for j in range(self.layers):
    #         o_i, r_i = self.retentions[j].forward_chunkwise(self.layer_norms_1[j](x_i), r_i_1s[j], i)
    #         y_i = o_i + x_i
    #         r_is.append(r_i)
    #         x_i = self.ffns[j](self.layer_norms_2[j](y_i)) + y_i
    #
    #     return x_i, r_is


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf

    This function is taken from the rwightman.
    It can be seen here:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py#L140
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class RetNet(nn.Module):
    def __init__(self, layers, hidden_dim, ffn_size, heads, double_v_dim=False):
        super(RetNet, self).__init__()
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.ffn_size = ffn_size
        self.heads = heads
        self.v_dim = hidden_dim * 2 if double_v_dim else hidden_dim
        self.norm = nn.LayerNorm(hidden_dim)
        self.retentions = nn.ModuleList([
            MultiScaleRetention(hidden_dim, heads, double_v_dim)
            for _ in range(layers)
        ])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, ffn_size),
                nn.GELU(),
                nn.Linear(ffn_size, hidden_dim)
            )
            for _ in range(layers)
        ])
        self.layer_norms_1 = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(layers)
        ])
        self.layer_norms_2 = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(layers)
        ])

    def forward(self, X):
        """
        X: (batch_size, sequence_length, hidden_size)
        """
        for i in range(self.layers):
            Y = self.retentions[i](self.layer_norms_1[i](X)) + X
            X = self.ffns[i](self.layer_norms_2[i](Y)) + Y
            X = self.norm(X)

        return X

    # def forward_recurrent(self, x_n, s_n_1s, n):
    #     """
    #     X: (batch_size, sequence_length, hidden_size)
    #     s_n_1s: list of lists of tensors of shape (batch_size, hidden_size // heads, hidden_size // heads)
    #
    #     """
    #     s_ns = []
    #     for i in range(self.layers):
    #         # list index out of range
    #         o_n, s_n = self.retentions[i].forward_recurrent(self.layer_norms_1[i](x_n), s_n_1s[i], n)
    #         y_n = o_n + x_n
    #         s_ns.append(s_n)
    #         x_n = self.ffns[i](self.layer_norms_2[i](y_n)) + y_n
    #
    #     return x_n, s_ns
    #
    # def forward_chunkwise(self, x_i, r_i_1s, i):
    #     """
    #     X: (batch_size, sequence_length, hidden_size)
    #     r_i_1s: list of lists of tensors of shape (batch_size, hidden_size // heads, hidden_size // heads)
    #
    #     """
    #     r_is = []
    #     for j in range(self.layers):
    #         o_i, r_i = self.retentions[j].forward_chunkwise(self.layer_norms_1[j](x_i), r_i_1s[j], i)
    #         y_i = o_i + x_i
    #         r_is.append(r_i)
    #         x_i = self.ffns[j](self.layer_norms_2[j](y_i)) + y_i
    #
    #     return x_i, r_is


if __name__ == '__main__':
    model = efr()
    model.eval()
    print(model)
    input = torch.randn((64, 1, 30, 13, 13))
    y ,cbrs= model(input)
    print(y.size())

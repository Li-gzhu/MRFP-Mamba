from itertools import repeat

import torch

from einops import rearrange
from torch.nn import init

from retention import MultiScaleRetention
import torch.nn.functional as F
from einops import rearrange, repeat

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
    def __init__(self, dim=40, expand_ratio=2, act_layer=nn.ReLU, drop_path_rate=0.0):
        super().__init__()
        self.pconv1 = PConv2_1(dim)
        self.conv1 = nn.Conv2d(dim, int(dim * expand_ratio), 1, bias=False)
        self.bn = nn.BatchNorm2d(int(dim * expand_ratio))
        self.act_layer = act_layer()
        self.conv2 = nn.Conv2d(int(dim * expand_ratio), dim, 1, bias=False)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.eca_layer = Eca_layer(dim)

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
        self.eca_layer = Eca_layer(dim)

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

    def __init__(self, k_size=3):
        super(Eca_layer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = torch.Tensor(x)
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
    def __init__(self, channels, faster_dim, act_layer=nn.ReLU):
        super().__init__()
        print(".....channels{}".format(channels))
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
    def __init__(self, channels,faster_dim,faster_dim1):#30,42,64
        super().__init__()
        self.embeding = Embeding(channels,faster_dim)
        self.mering1 = Mering(faster_dim, faster_dim1)
        # self.mering2 = Mering(faster_dim*2)
        # self.mering3 = Mering(faster_dim*4)
        self.FasterNetBlock2_1 =FasterNetBlock2_1(faster_dim)#
        self.FasterNetBlock2_2 =FasterNetBlock2_2(faster_dim1)


        # self.FasterNetBlock1 = FasterNetBlock1(faster_dim)
        # self.FasterNetBlock2 = FasterNetBlock1(faster_dim * 2)
        # self.FasterNetBlock3 = FasterNetBlock1(faster_dim * 4)
        # self.FasterNetBlock4 = FasterNetBlock1(faster_dim * 8)


    def forward(self, x):  # x 64,30,13,13
        x = self.embeding(x)# 64，40，11，11
        #2层串行
        # x = self.FasterNetBlock2_1(x)
        x = self.mering1(x)
        # x = self.FasterNetBlock2_2(x)

        #faster并行的方式
        # x1 = self.FasterNetBlock2_1(x)#利用比较有效的波段
        # x2 = self.FasterNetBlock2_2(x)
        # x = x1+ x2

        return x









#        model = ViT(dim=128, image_size=15, patch_size=3, depth=6, heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1, num_classes=n_classes, channels=n_bands,)

class EFT(nn.Module):
    def __init__(self,faster_dim=40, hidden_dim=64,  channels=30,
                    num_classes=1,   depth=1,dim =64,
                 heads=8, mlp_dim=8, dropout=0, emb_dropout=0):
        super().__init__()
        # 添加

        self.fasternet = FasterNet(channels, faster_dim, hidden_dim)#30,42,
        self.hidden_dim = hidden_dim
        self.ffn_size = hidden_dim * 3
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        # self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)depth=1,dim =64,
        #                  heads=8, mlp_dim=8, dropout=0.1, emb_dropout=0.1

        self.dropout = nn.Dropout(emb_dropout)
        # self.nn1 = nn.Linear((hidden_dim), num_classes)  #
        # torch.nn.init.xavier_uniform_(self.nn1.weight)
        # torch.nn.init.normal_(self.nn1.bias, std=1e-6)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 121 + 1, dim))

    def forward(self, X):
        """
        X: (batch_size, sequence_length, hidden_size)
        """
        X = torch.squeeze(X)  # 64,30,13,13
        X = self.fasternet(X)  #64,320,5,5
        X = rearrange(X, 'b c h w -> b (h w) c')
        b, n, _ = X.shape
        # print("....b   {}".format(b))
        # print("....n   {}".format(n))

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, X), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        X = self.transformer(X)  # main game
        X = X.mean(dim=1)
        X = self.mlp_head(X)
        return X


class PConv1(nn.Module):
    def __init__(self, dim, kernel_size=3, n_div=4, ):
        super().__init__()
        self.dim_conv = dim // n_div
        self.conv = nn.Conv2d(self.dim_conv, self.dim_conv, kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)

    def forward(self, x):
        x1, x2, x3, x4 = torch.split(x, [self.dim_conv, self.dim_conv, self.dim_conv, self.dim_conv], dim=1)
        x1 = self.conv(x1)
        x = torch.concat([x1, x2, x3, x4], dim=1)

        return x


class PConv2(nn.Module):
    def __init__(self, dim, kernel_size=3, n_div=4, ):
        super().__init__()
        self.dim_conv = dim // n_div
        self.conv = nn.Conv2d(self.dim_conv, self.dim_conv, kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)

    def forward(self, x):
        x1, x2, x3, x4 = torch.split(x, [self.dim_conv, self.dim_conv, self.dim_conv, self.dim_conv], dim=1)
        x2 = self.conv(x2)
        x = torch.concat([x1, x2, x3, x4], dim=1)

        return x


class PConv3(nn.Module):
    def __init__(self, dim, kernel_size=3, n_div=4, ):
        super().__init__()
        self.dim_conv = dim // n_div
        self.conv = nn.Conv2d(self.dim_conv, self.dim_conv, kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)

    def forward(self, x):
        x1, x2, x3, x4 = torch.split(x, [self.dim_conv, self.dim_conv, self.dim_conv, self.dim_conv], dim=1)
        x3 = self.conv(x3)
        x = torch.concat([x1, x2, x3, x4], dim=1)

        return x


class PConv4(nn.Module):
    def __init__(self, dim, kernel_size=3, n_div=4, ):
        super().__init__()
        self.dim_conv = dim // n_div
        self.conv = nn.Conv2d(self.dim_conv, self.dim_conv, kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)

    def forward(self, x):
        x1, x2, x3, x4 = torch.split(x, [self.dim_conv, self.dim_conv, self.dim_conv, self.dim_conv], dim=1)
        x4 = self.conv(x4)
        x = torch.concat([x1, x2, x3, x4], dim=1)

        return x


class FasterNetBlock1(nn.Module):
    def __init__(self, dim=40, expand_ratio=2, act_layer=nn.ReLU, drop_path_rate=0.0):
        super().__init__()
        self.pconv1 = PConv1(dim)
        self.conv1 = nn.Conv2d(dim, int(dim * expand_ratio), 1, bias=False)
        self.bn = nn.BatchNorm2d(int(dim * expand_ratio))
        self.act_layer = act_layer()
        self.conv2 = nn.Conv2d(int(dim * expand_ratio), dim, 1, bias=False)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.eca_layer = Eca_layer(dim)

    def forward(self, x):
        residual = x  #64,40,11
        x = self.eca_layer(x)
        x = self.pconv1(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.act_layer(x)
        x = self.conv2(x)
        x = residual + self.drop_path(x)
        return x


class FasterNetBlock2(nn.Module):
    def __init__(self, dim=80, expand_ratio=2, act_layer=nn.ReLU, drop_path_rate=0.0):
        super().__init__()
        self.pconv2 = PConv2(dim)
        self.conv1 = nn.Conv2d(dim, int(dim * expand_ratio), 1, bias=False)
        self.bn = nn.BatchNorm2d(int(dim * expand_ratio))
        self.act_layer = act_layer()
        self.conv2 = nn.Conv2d(int(dim * expand_ratio), dim, 1, bias=False)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.eca_layer = Eca_layer(dim)

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


class FasterNetBlock3(nn.Module):
    def __init__(self, dim=160, expand_ratio=2, act_layer=nn.ReLU, drop_path_rate=0.0):
        super().__init__()
        self.pconv3 = PConv3(dim)
        self.conv1 = nn.Conv2d(dim, int(dim * expand_ratio), 1, bias=False)
        self.bn = nn.BatchNorm2d(int(dim * expand_ratio))
        self.act_layer = act_layer()
        self.conv2 = nn.Conv2d(int(dim * expand_ratio), dim, 1, bias=False)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.eca_layer = Eca_layer(dim)

    def forward(self, x):
        residual = x
        x = self.eca_layer(x)
        x = self.pconv3(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.act_layer(x)
        x = self.conv2(x)
        x = residual + self.drop_path(x)
        return x


class FasterNetBlock4(nn.Module):
    def __init__(self, dim=320, expand_ratio=2, act_layer=nn.ReLU, drop_path_rate=0.0):
        super().__init__()
        self.pconv4 = PConv4(dim)
        self.conv1 = nn.Conv2d(dim, int(dim * expand_ratio), 1, bias=False)
        self.bn = nn.BatchNorm2d(int(dim * expand_ratio))
        self.act_layer = act_layer()
        self.conv2 = nn.Conv2d(int(dim * expand_ratio), dim, 1, bias=False)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.eca_layer = Eca_layer(dim)

    def forward(self, x):
        residual = x
        x = self.eca_layer(x)
        x = self.pconv4(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.act_layer(x)
        x = self.conv2(x)

        x = residual + self.drop_path(x)
        return x

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


def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class LayerNormalize(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# 等于 FeedForward
class MLP_Block(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):

    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(dim)

        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)  # Wq,Wk,Wv for each vector, thats why *3
        # torch.nn.init.xavier_uniform_(self.to_qkv.weight)
        # torch.nn.init.zeros_(self.to_qkv.bias)

        self.nn1 = nn.Linear(dim, dim)
        # torch.nn.init.xavier_uniform_(self.nn1.weight)
        # torch.nn.init.zeros_(self.nn1.bias)
        self.do1 = nn.Dropout(dropout)

    def forward(self, x, mask=None):

        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)  # gets q = Q = Wq matmul x1, k = Wk mm x2, v = Wv mm x3
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)  # split into multi head attentions

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)  # follow the softmax,q,d,v equation in the paper

        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # product of v times whatever inside softmax
        out = rearrange(out, 'b h n d -> b n (h d)')  # concat heads into one matrix, ready for next encoder block
        out = self.nn1(out)
        out = self.do1(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(LayerNormalize(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(LayerNormalize(dim, MLP_Block(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attention, mlp in self.layers:
            x = attention(x, mask=mask)  # go to attention
            x = mlp(x)  # go to MLP_Block
        return x

if __name__ == '__main__':
    model = EFT()
    model.eval()
    print(model)
    input = torch.randn((64, 1, 30, 13, 13))
    y = model(input)
    print(y.size())

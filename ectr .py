import torch

from einops import rearrange
from retention import MultiScaleRetention

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
        self.conv1 = nn.Conv2d(dim, int(dim * expand_ratio), 3, bias=False)
        self.bn = nn.BatchNorm2d(int(dim * expand_ratio))
        self.act_layer = act_layer()
        self.conv2 = nn.Conv2d(int(dim * expand_ratio), dim, 1, bias=False)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.eca_layer = Eca_layer(dim)

    def forward(self, x):
        residual = x
        x = self.eca_layer(x)
        #x = self.pconv1(x)
        # x = self.conv1(x)
        # x = self.bn(x)
        # x = self.act_layer(x)
        #x = self.conv2(x)
        #x = residual + self.drop_path(x)
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
        self.conv = nn.Conv1d(1, 1, kernel_size=5, padding=(5 - 1) // 2, bias=False)
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
    def __init__(self, channels=None, faster_dim=42, act_layer=nn.ReLU):
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
        self.embeding = Embeding(channels)
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
        x = self.FasterNetBlock2_1(x)
        x = self.mering1(x)
        x = self.FasterNetBlock2_2(x)

        #faster并行的方式
        # x1 = self.FasterNetBlock2_1(x)#利用比较有效的波段
        # x2 = self.FasterNetBlock2_2(x)
        # x = x1+ x2

        return x





class Tokenizer(nn.Module):
    def __init__(self, token_L, token_cT):
        super(Tokenizer, self).__init__()
        # self.L = L  # num_token
        # self.cT = cT  # dim
        self.token_wA = nn.Parameter(torch.empty(1, token_L, token_cT),  # 1，4，80
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, token_cT, token_cT),  # 1，64，64
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wV)

        # self.pos_embedding = nn.Parameter(torch.empty(1, (self.L + 1), self.cT))  # 1，4+1，64
        # torch.nn.init.normal_(self.pos_embedding, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_cT))  # 1，1，64

    def forward(self, X):
        wa = rearrange(self.token_wA, 'b h w -> b w h')  # Transpose  wa  #1，64，4；
        A = torch.einsum('bij,bjk->bik', X, wa)  ##x  81*64； a  81*4；
        A = rearrange(A, 'b h w -> b w h')  # Transpose 转置  4  8
        A = A.softmax(dim=-1)

        VV = torch.einsum('bij,bjk->bik', X, self.token_wV)  # token_wV  #1，64，64 ； vv 81*64
        X = torch.einsum('bij,bjk->bik', A, VV)  # T 64,4,160
        return X



#        model = ViT(dim=128, image_size=15, patch_size=3, depth=6, heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1, num_classes=n_classes, channels=n_bands,)

class ECTR(nn.Module):
    def __init__(self, layers_ret=1, image_size=None, patch_size=None,
                 hidden_dim=None, ffn_size=None, heads=None, dim1=None,
                 dim2=None, num_classes=None, channels=None, faster_dim=None,
                 token_L=None, double_v_dim=False):
        super().__init__()
        # 添加

        self.fasternet = FasterNet(channels, faster_dim, hidden_dim)#30,42,


        self.list_retnet = {"layers": layers_ret, "hidden_dim": hidden_dim, "ffn_size": ffn_size, "heads": heads,
                            "dim1": dim1, "num_classes": num_classes}

        self.layers = layers_ret
        self.hidden_dim = hidden_dim
        self.ffn_size = hidden_dim * 3
        #self.heads = heads
        self.v_dim = hidden_dim * 2 if double_v_dim else hidden_dim

        self.retentions = nn.ModuleList([
            MultiScaleRetention(hidden_dim, heads, double_v_dim)
            for _ in range(layers_ret)
        ])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, ffn_size),
                nn.GELU(),
                nn.Linear(ffn_size, hidden_dim)
            )
            for _ in range(layers_ret)
        ])
        self.layer_norms_1 = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(layers_ret)
        ])
        self.layer_norms_2 = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(layers_ret)
        ])

        self.nn1 = nn.Linear(dim1, num_classes)  #
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)
        self.flatten = nn.Flatten(start_dim=1)
        self.tokenizer = Tokenizer(token_L, hidden_dim,)
        self.to_cls_token = nn.Identity()

        # self.nn2 = nn.Linear(dim2, num_classes)
        # torch.nn.init.xavier_uniform_(self.nn2.weight)
        # torch.nn.init.normal_(self.nn2.bias, std=1e-6)

    def forward(self, X):
        """
        X: (batch_size, sequence_length, hidden_size)
        """
        X = torch.squeeze(X)  # 64,30,13,13

        # X = self.conv9size(X)
        X = self.fasternet(X)  #64,320,5,5

        X = rearrange(X, 'b c h w -> b (h w) c')

        X = self.tokenizer(X)  # 64,81,160

        # X = torch.transpose(X, -1, -2)  #X(64,64,64)
        for i in range(self.layers):
            Y = self.retentions[i](self.layer_norms_1[i](X)) + X  # Y(64,81,64)

            X = self.ffns[i](self.layer_norms_2[i](Y)) + Y  # X(64,81,64)
            if i == (self.layers - 1):
                X = self.flatten(X)
                # X = self.to_cls_token(X[:, 0])  # 64,160  [:, 0]表示获取所有行（样本）的第一个列元素。
                X = self.nn1(X)
            # X = self.nn2(X)
        out = self.list_retnet
        return X

    def forward_recurrent(self, x_n, s_n_1s, n):
        """
        X: (batch_size, sequence_length, hidden_size)
        s_n_1s: list of lists of tensors of shape (batch_size, hidden_size // heads, hidden_size // heads)

        """
        s_ns = []
        for i in range(self.layers):
            # list index out of range
            o_n, s_n = self.retentions[i].forward_recurrent(self.layer_norms_1[i](x_n), s_n_1s[i], n)
            y_n = o_n + x_n
            s_ns.append(s_n)
            x_n = self.ffns[i](self.layer_norms_2[i](y_n)) + y_n

        return x_n, s_ns

    def forward_chunkwise(self, x_i, r_i_1s, i):
        """
        X: (batch_size, sequence_length, hidden_size)
        r_i_1s: list of lists of tensors of shape (batch_size, hidden_size // heads, hidden_size // heads)

        """
        r_is = []
        for j in range(self.layers):
            o_i, r_i = self.retentions[j].forward_chunkwise(self.layer_norms_1[j](x_i), r_i_1s[j], i)
            y_i = o_i + x_i
            r_is.append(r_i)
            x_i = self.ffns[j](self.layer_norms_2[j](y_i)) + y_i

        return x_i, r_is


# 3d,2d卷积

# class Convnet(nn.Module):
#     def __init__(self, in_channels=1):
#         super(Convnet, self).__init__()
#         # self.L = num_tokens
#         # self.cT = dim
#         self.conv3d_features = nn.Sequential(
#             nn.Conv3d(in_channels, out_channels=8, kernel_size=(3, 3, 3)),#通过卷积运算生成8个11 × 11 × 28的特征立方体
#             nn.BatchNorm3d(8),
#             nn.ReLU(),
#         )#x = rearrange(x, 'b c h w y -> b (c h) w y')
# #平面化成一个1-D特征向量，得到大小为1×81的64个向量
#         self.conv2d_features = nn.Sequential(
#             nn.Conv2d(in_channels=8*28, out_channels=64, kernel_size=(3, 3)),#将八个特征立方体重新排列，生成一个11 × 11 × 224的特征立方体。作为2d卷积的输入。得到64*9*9；
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#         )
#
#
#         # self.conv2d_features2 = nn.Sequential(
#         #     nn.Conv2d(in_channels=112, out_channels=64, kernel_size=(2, 2)),#得到64*8*8；
#         #     nn.BatchNorm2d(64),
#         #     nn.ReLU(),
#         # )
#
#     def forward(self, X, mask=None):
#         X = self.conv3d_features(X)  # patch is 13×13×30,8颗3×3×3立方核,通过卷积运算生成8个11 × 11 × 28的特征立方体
#         X = rearrange(X, 'b c h w y -> b (c h) w y')
#         X = self.conv2d_features(X)  # 64个3 × 3平面核，得到64个大小为9×9。
#        # X = self.conv2d_features2(X)  #64*8*8
#         X = rearrange(X, 'b c h w -> b (h w) c') #64,81,64
#         return X

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


if __name__ == '__main__':
    model = EftR()
    model.eval()
    print(model)
    input = torch.randn((64, 1, 30, 13, 13))
    y ,cbrs= model(input)
    print(y.size())

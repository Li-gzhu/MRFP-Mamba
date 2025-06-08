import math
import torch
import torch.nn as nn
from einops import rearrange

from retention import MultiScaleRetention

num_classes=9
class ret3(nn.Module):
    def __init__(self, patch_size, channels=103, num_classes=9, num_stages=3,
                 L=4, cT=40,
                  heads=[1, 1, 1],  ffn=[3,3,3],layers=[1, 1, 1]):
        super().__init__()
        s = [patch_size-2,patch_size-2,patch_size-2,]
        n_groups = [32]
        new_bands = math.ceil(channels / n_groups[0]) * n_groups[0]
        hidden_dim = [new_bands, new_bands, new_bands]
        print(hidden_dim)
        print(s)

        # new_bands1 = math.ceil((channels+32) / n_groups[0]) * n_groups[0]
        # new_bands2 = math.ceil((channels+64) / n_groups[0]) * n_groups[0]
        self.num_stages = num_stages
        self.Embeding1 = Embeding1(channels,new_bands)
        self.FirstConv = FirstConv(channels,hidden_dim[0])
        self.FirstConv1 = FirstConv(hidden_dim[0],hidden_dim[1])
        self.FirstConv2 = FirstConv(hidden_dim[1],hidden_dim[2])

        # self.FasterNetBlock3_1 = FasterNetBlock3_1(channels,hidden_dim[0])
        self.FasterNetBlock2_1 = FasterNetBlock2_1(hidden_dim[0],hidden_dim[1])
        self.FasterNetBlock2_2 = FasterNetBlock2_2(hidden_dim[1],hidden_dim[2])
        # for i in range(num_stages):
        #     patch_embed = Embeding1(
        #         in_feature_map_size=img_size_list[i],
        #         in_chans=new_bands if i == 0 else embed_dims[i - 1],
        #         embed_dim=embed_dims[i],
        #         n_groups=n_groups[i]
        #     )
        # self.conv0 = nn.ModuleList([FasterNetBlock2_1,FasterNetBlock2_2])
        # self.convList = [self.FirstConv, self.FasterNetBlock2_1, self.FasterNetBlock2_2]
        self.convList = [self.Embeding1, self.FasterNetBlock2_1, self.FasterNetBlock2_2]

        # self.to_image = nn.ModuleList(To_Image(s[i])for i in range(num_stages-1))
        self.retnets = nn.ModuleList([RetNet(
            layers[i],  hidden_dim[i], ffn[i], heads[i], s[i], double_v_dim=False)for i in range(num_stages)])
        # for i in range(num_stages):
        #     self.norm = nn.LayerNorm(hidden_dim[i])

        self.head = nn.Linear(hidden_dim[-1], num_classes)

    def forward(self, x):
        # (bs, 1, n_bands, patch size (ps, of HSI), ps)
        # x = self.pad(x).squeeze(dim=1)
        B = x.shape[0]
        x = torch.squeeze(x)
        for i in range(self.num_stages):
            # print("....  i  {}".format(i))
            x = self.convList[i](x)
            # print("....convList[i]{}    ".format(x.shape))

            x, s = self.retnets[i](x)
            # print("....retnets[i]{}".format(x.shape))
            # print("....s  {}".format(s))

            if i != self.num_stages - 1:
                x = x.reshape(B, s, s, -1).permute(0, 3, 1, 2).contiguous()
                # print("....  i  {}".format(i))
                # print("....To_Image{}".format(x.shape))
        x = x.mean(dim=1)
        x = self.head(x)
        return x

            # patch_embed = getattr(self, f"patch_embed{i + 1}")
            # block = getattr(self, f"block{i + 1}")
            # norm = getattr(self, f"norm{i + 1}")
            #
            # x, s = patch_embed(x)  # s = feature map size after patch embedding
            # for blk in block:
            #     x = blk(x)
            #
            # x = norm(x)
            #
            # if i != self.num_stages - 1:
            #     x = x.reshape(B, s, s, -1).permute(0, 3, 1, 2).contiguous()
        

    # def forward(self, x):
    #     for i in range(self.num_stages):
    #
    #         print("....x.shape{}".format(x.shape))
    #         x = self.forward_features(x)
    #         print("....forward_features   x.shape{}".format(x.shape))
    #
    #         x = x.mean(dim=1)
    #         x = self.head(x)


# class To_Image(nn.Module):
#     def __init__(self,s):
#         super().__init__()
#         self.s = s
#     def forward(self, x):
#         self.s = self.s
#         print("....To_Image   x{}".format(x.shape))
#         B = x.shape[0]
#         x = x.reshape(B, self.s, self.s, -1).transpose(0, 3, 1, 2)
#         # x = rearrange(x,'b (a a) c -> b a a w')
#         print("....after   To_Image   x{}".format(x.shape))
#         return x


class RetNet(nn.Module):
    def __init__(self, layers, hidden_dim, ffn, heads, s, double_v_dim=False):
        super(RetNet, self).__init__()
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.ffn = ffn
        self.heads = heads
        self.v_dim = hidden_dim * 2 if double_v_dim else hidden_dim
        # print(".....hidden_dim{}".format(hidden_dim))
        self.retentions = nn.ModuleList([
            MultiScaleRetention(hidden_dim, heads, double_v_dim)
            for _ in range(layers)
        ])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, ffn*hidden_dim),
                nn.GELU(),
                nn.Linear(ffn*hidden_dim, hidden_dim)
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

        self.norm = nn.LayerNorm(hidden_dim)
        self.s = s
    def forward(self, X):
        """
        X: (batch_size, sequence_length, hidden_size)
        """
        s = self.s
        for i in range(self.layers):
            Y = self.retentions[i](self.layer_norms_1[i](X)) + X
            X = self.ffns[i](self.layer_norms_2[i](Y)) + Y
            X = self.norm(X)
        return X, s

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


class Tokenizer(nn.Module):
    def __init__(self, L=4, cT=40):
        super(Tokenizer, self).__init__()
        self.L = L  # num_token
        self.cT = cT  # dim
        self.token_wA = nn.Parameter(torch.empty(1, self.L, self.cT),  # 1，4，80
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wA)
        self.token_wV = nn.Parameter(torch.empty(1, self.cT, self.cT),  # 1，64，64
                                     requires_grad=True)  # Tokenization parameters
        torch.nn.init.xavier_normal_(self.token_wV)

    def forward(self, X):
        wa = rearrange(self.token_wA, 'b h w -> b w h')  # Transpose  wa  #1，64，4；
        A = torch.einsum('bij,bjk->bik', X, wa)  ##x  81*64； a  81*4；
        A = rearrange(A, 'b h w -> b w h')  # Transpose 转置  4  8
        A = A.softmax(dim=-1)

        VV = torch.einsum('bij,bjk->bik', X, self.token_wV)  # token_wV  #1，64，64 ； vv 81*64
        X = torch.einsum('bij,bjk->bik', A, VV)  # T 64,4,160
        return X

class Eca_layer(nn.Module):
    """Constructs a ECA module.*
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, k_size=4):
        super(Eca_layer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=5, padding=(5-1) // 2, bias=False)
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


class Embeding1(nn.Module):
    def __init__(self, in_channel, new_bands, act_layer=nn.ReLU):
        super().__init__()
        # n_groups=[32]
        # new_bands = math.ceil(in_channel / n_groups[0]) * n_groups[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channel, new_bands, 3, stride=1, bias=False),
            nn.BatchNorm2d(new_bands),
            act_layer()
        )

    def forward(self, x):
        x = self.stem(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class Embeding2(nn.Module):
    def __init__(self, in_channel=30, act_layer=nn.ReLU):
        super().__init__()
        n_groups=[32]
        new_bands = math.ceil((in_channel+32) / n_groups[0]) * n_groups[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channel, new_bands, 3, stride=1, bias=False),
            nn.BatchNorm2d(new_bands),
            act_layer()
        )

    def forward(self, x):
        x = self.stem(x)
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


class FirstConv(nn.Module):
    def __init__(self, embed_dim, embed_dim1, act_layer=nn.ReLU ):
        super().__init__()
        # self.up = nn.Sequential(
        #                 nn.Conv2d(embed_dim, embed_dim1, 3, stride=1, bias=False),
        #                 nn.BatchNorm2d(embed_dim1),
        #                 act_layer()
        #             )
        self.proj = nn.Conv2d(embed_dim, embed_dim1, kernel_size=3, stride=1, padding=1,)
        self.batch_norm = nn.BatchNorm2d(embed_dim1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x = self.up(x)
        x = self.proj(x)
        x = self.relu(self.batch_norm(x))
        x = x.flatten(2).transpose(1, 2)
        return x

class PConv3_1(nn.Module):
    def __init__(self, dim, kernel_size=3, n_div=3, ):
        super().__init__()
        self.dim_conv = dim // n_div
        self.conv = nn.Conv2d(self.dim_conv, self.dim_conv, kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)

    def forward(self, x):
        x1, x2, x3 = torch.split(x, [self.dim_conv, self.dim_conv,self.dim_conv], dim=1)
        x1 = self.conv(x1)
        x = torch.concat([x1, x2, x3], dim=1)

        return x

class PConv3_2(nn.Module):
    def __init__(self, dim, kernel_size=3, n_div=3, ):
        super().__init__()
        self.dim_conv = dim // n_div
        self.conv = nn.Conv2d(self.dim_conv, self.dim_conv, kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)

    def forward(self, x):
        x1, x2, x3 = torch.split(x, [self.dim_conv, self.dim_conv,self.dim_conv], dim=1)
        x2 = self.conv(x2)
        x = torch.concat([x1, x2, x3], dim=1)

        return x

class PConv3_3(nn.Module):
    def __init__(self, dim, kernel_size=3, n_div=3, ):
        super().__init__()
        self.dim_conv = dim // n_div
        self.conv = nn.Conv2d(self.dim_conv, self.dim_conv, kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)

    def forward(self, x):
        x1, x2, x3 = torch.split(x, [self.dim_conv, self.dim_conv,self.dim_conv], dim=1)
        x3 = self.conv(x3)
        x = torch.concat([x1, x2, x3], dim=1)

        return x


class FasterNetBlock3_1(nn.Module):
    def __init__(self, dim=40, expand_ratio=2, act_layer=nn.ReLU, drop_path_rate=0.0):
        super().__init__()
        self.pconv1 = PConv3_1(dim)
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
        x = x.flatten(2).transpose(1, 2)

        return x


class FasterNetBlock3_2(nn.Module):
    def __init__(self, dim=40, expand_ratio=2, act_layer=nn.ReLU, drop_path_rate=0.0):
        super().__init__()
        self.pconv2 = PConv3_2(dim)
        self.conv1 = nn.Conv2d(dim, int(dim * expand_ratio), 1, bias=False)
        self.bn = nn.BatchNorm2d(int(dim * expand_ratio))
        self.act_layer = act_layer()
        self.conv2 = nn.Conv2d(int(dim * expand_ratio), dim, 1, bias=False)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.eca_layer = Eca_layer(dim)

    def forward(self, x):
        residual = x  #64,40,11
        x = self.eca_layer(x)
        x = self.pconv2(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.act_layer(x)
        x = self.conv2(x)
        x = residual + self.drop_path(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class FasterNetBlock3_3(nn.Module):
    def __init__(self, dim=40, expand_ratio=2, act_layer=nn.ReLU, drop_path_rate=0.0):
        super().__init__()
        self.pconv3 = PConv3_3(dim)
        self.conv1 = nn.Conv2d(dim, int(dim * expand_ratio), 1, bias=False)
        self.bn = nn.BatchNorm2d(int(dim * expand_ratio))
        self.act_layer = act_layer()
        self.conv2 = nn.Conv2d(int(dim * expand_ratio), dim, 1, bias=False)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.eca_layer = Eca_layer(dim)

    def forward(self, x):
        residual = x  #64,40,11
        x = self.eca_layer(x)
        x = self.pconv3(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.act_layer(x)
        x = self.conv2(x)
        x = residual + self.drop_path(x)
        x = x.flatten(2).transpose(1, 2)

        return x

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
        residual = x  #64,40,11
        x = self.eca_layer(x)
        x = self.pconv1(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.act_layer(x)
        x = self.conv2(x)
        x = residual + self.drop_path(x)
        x = x.flatten(2).transpose(1, 2)

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
        x = x.flatten(2).transpose(1, 2)

        return x

class Faster1(nn.Module):
    def __init__(self, dim=80, expand_ratio=2, act_layer=nn.ReLU, drop_path_rate=0.0):
        super().__init__()
        self.Embeding1 = Embeding1()
        self.FasterNetBlock2_1


if __name__ == "__main__":
    model = ret3()
    model.eval()
    print(model)
    input = torch.randn((64, 1, 103, 13, 13))
    y = model(input)





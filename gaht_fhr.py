import math
import torch
import torch.nn as nn
from retention import MultiScaleRetention


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        input: (B, N, C)
        B = Batch size, N = patch_size * patch_size, C = dimension hidden_features and out_features
        output: (B, N, C)
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=16, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        """
        input: (B, N, C)
        B = Batch size, N = patch_size * patch_size, C = dimension for attention
        output: (B, N, C)
        """
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class GroupedPixelEmbedding(nn.Module):
    def __init__(self, in_feature_map_size=7, in_chans=3, embed_dim=128, n_groups=1):
        super().__init__()
        self.ifm_size = in_feature_map_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1, groups=n_groups)
        self.batch_norm = nn.BatchNorm2d(embed_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        input: (B, in_chans, in_feature_map_size, in_feature_map_size)
        output: (B, (after_feature_map_size x after_feature_map_size-2), embed_dim = C)
        """
        # print("...before proj shape{}   ".format(x.shape))
        x = self.proj(x)
        x = self.relu(self.batch_norm(x))
        # print("...after proj shape{}   ".format(x.shape))

        x = x.flatten(2).transpose(1, 2)
        # print("...after flatten shape{}   ".format(x.shape))

        after_feature_map_size = self.ifm_size  
        
        return x, after_feature_map_size


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class gaht(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000, num_stages=3, 
                n_groups=[16, 16, 16], embed_dims=[256, 128, 64], num_heads=[8, 4, 2], mlp_ratios=[1, 1, 1], depths=[1, 1, 1],drop_path_rate=0.1):
        super().__init__()

        self.num_stages = num_stages
        
        new_bands = math.ceil(in_chans / n_groups[0]) * n_groups[0]
        self.pad = nn.ReplicationPad3d((0, 0, 0, 0, 0, new_bands - in_chans))
        # self.retnet1 = RetNet( layers_ret1, hidden_dim, ffn1*hidden_dim, heads1, double_v_dim=False)
        # dpr = [x.item()
        #        for x in torch.linspace(0, drop_path_rate, sum(depths)-1)]
        # print("...dpr {}".format(dpr))
        for i in range(num_stages):

            patch_embed = GroupedPixelEmbedding(
                in_feature_map_size=img_size,
                in_chans=new_bands if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i],
                n_groups=n_groups[i]
            )

            # faster_block = Faster_Block(i, embed_dim=new_bands if i == 0 else embed_dims[i - 1],)

            # faster_block = Faster_Block(i, embed_dim=embed_dims[i],)
            faster_block2 = Faster_Block2(i, embed_dim=embed_dims[i],)

            # ret_block = nn.ModuleList([RetNet(layers=1, hidden_dim=embed_dims[i], ffn_size=2, heads = num_heads[i], )
            #                            for j in range(depths[i])])


            block = nn.ModuleList([Block(
                dim=embed_dims[i],
                num_heads=num_heads[i],
                mlp_ratio=mlp_ratios[i],
                drop=0.,
                attn_drop=0.) for j in range(depths[i])])

            norm = nn.LayerNorm(embed_dims[i])

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            # setattr(self, f"faster_block{i + 1}", faster_block)
            setattr(self, f"faster_block2{i + 1}", faster_block2)

            # setattr(self, f"ret_block{i + 1}", ret_block)

            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)
        
        self.head = nn.Linear(embed_dims[-1], num_classes)  # 只有pvt时的Head

    def forward_features(self, x):
        # (bs, 1, n_bands, patch size (ps, of HSI), ps)
        x = self.pad(x).squeeze(dim=1)
        B = x.shape[0]

        for i in range(self.num_stages):
            # faster_block = getattr(self, f"faster_block{i + 1}")
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            # faster_block = getattr(self, f"faster_block{i + 1}")
            faster_block2 = getattr(self, f"faster_block2{i + 1}")
            # ret_block = getattr(self, f"ret_block{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")



            # x = faster_block(x)
            x, s = patch_embed(x)  # s = feature map size after patch embedding
            # if i == 0 :
            #     x = x.flatten(2).transpose(1, 2)
            #
            # # for blk in ret_block:
            # else:
            # x = faster_block2(x)
            # x = faster_block(x)

            for blk in block:

                x = blk(x)

            # for blk in block:
            #     x = blk(x)
            
            x = norm(x)
            if i != self.num_stages - 1:
                x = x.reshape(B, s, s, -1).permute(0, 3, 1, 2).contiguous()
                x = faster_block2(x).contiguous()

        return x
    
    def forward(self, x):
        # print("....x.shape{}".format(x.shape))
        x = self.forward_features(x)
        # print("....forward_features   x.shape{}".format(x.shape))

        x = x.mean(dim=1)
        x = self.head(x)
        return x


class Faster_Block(nn.Module):
    def __init__(self,  i,embed_dim,):#30,42,64
        super().__init__()
        # print("......embed_dim  {}".format(embed_dim))
        # print("......drop_path_rate  {}".format(drop_path_rate))

        self.i = i
        self.FasterNetBlock3_1 =FasterNetBlock3_1(embed_dim,)#
        self.FasterNetBlock3_2 =FasterNetBlock3_2(embed_dim,)
        self.FasterNetBlock3_3 =FasterNetBlock3_3(embed_dim,)
        self.faster = [self.FasterNetBlock3_1,self.FasterNetBlock3_2,self.FasterNetBlock3_3]

    def forward(self, x):  # x 64,30,13,13
        x = self.faster[self.i](x)# 64，40，11，11
        # x = x.flatten(2).transpose(1, 2)

        return x


class Faster_Block2(nn.Module):
    def __init__(self,  i,embed_dim,):#30,42,64
        super().__init__()
        # print("......embed_dim  {}".format(embed_dim))
        # print("......drop_path_rate  {}".format(drop_path_rate))

        self.i = i
        self.FasterNetBlock2_1 =FasterNetBlock2_1(embed_dim,)#
        self.FasterNetBlock2_2 =FasterNetBlock2_2(embed_dim,)
        self.faster = [self.FasterNetBlock2_1,self.FasterNetBlock2_2]

    def forward(self, x):  # x 64,30,13,13
        x = self.faster[self.i](x)# 64，40，11，11
        # x = x.flatten(2).transpose(1, 2)

        return x



class PConv3_1(nn.Module):
    def __init__(self, dim, kernel_size=3, n_div=3, ):
        super().__init__()
        self.dim_conv = dim // n_div
        self.dim_last = dim - self.dim_conv*2
        self.conv = nn.Conv2d(self.dim_conv, self.dim_conv, kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)

    def forward(self, x):

        x1, x2, x3 = torch.split(x, [self.dim_conv, self.dim_conv, self.dim_last], dim=1)
        x1 = self.conv(x1)
        x = torch.concat([x1, x2, x3], dim=1)

        return x


class PConv3_2(nn.Module):
    def __init__(self, dim, kernel_size=3, n_div=3, ):
        super().__init__()
        self.dim_conv = dim // n_div
        self.dim_last = dim - self.dim_conv*2

        self.conv = nn.Conv2d(self.dim_conv, self.dim_conv, kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)

    def forward(self, x):
        x1, x2, x3 = torch.split(x, [self.dim_conv, self.dim_conv, self.dim_last], dim=1)
        x2 = self.conv(x2)
        x = torch.concat([x1, x2, x3], dim=1)

        return x


class PConv3_3(nn.Module):
    def __init__(self, dim, kernel_size=3, n_div=3, ):
        super().__init__()
        self.dim_conv = dim // n_div
        self.dim_last = dim - self.dim_conv*2

        self.conv = nn.Conv2d(self.dim_last, self.dim_last, kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)

    def forward(self, x):
        x1, x2, x3 = torch.split(x, [self.dim_conv, self.dim_conv, self.dim_last], dim=1)
        x3 = self.conv(x3)
        x = torch.concat([x1, x2, x3], dim=1)
        return x

class FasterNetBlock3_1(nn.Module):
    def __init__(self, dim, expand_ratio=2, act_layer=nn.ReLU, drop_path_rate=0.0):
        super().__init__()
        self.pconv1 = PConv3_1(dim)
        # print(".....dim {}".format(dim))
        self.conv1 = nn.Conv2d(dim, int(dim * expand_ratio), 1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(int(dim * expand_ratio))
        self.act_layer = act_layer()
        self.conv2 = nn.Conv2d(int(dim * expand_ratio), dim, 1, stride=1, bias=False)
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

class FasterNetBlock3_2(nn.Module):
    def __init__(self, dim=80, expand_ratio=2, act_layer=nn.ReLU, drop_path_rate=0.0):
        super().__init__()
        self.pconv2 = PConv3_2(dim)
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

class FasterNetBlock3_3(nn.Module):
    def __init__(self, dim=80, expand_ratio=2, act_layer=nn.ReLU, drop_path_rate=0.0):
        super().__init__()
        self.pconv2 = PConv3_3(dim)
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
        residual1 = x + residual
        x = self.pconv1(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.act_layer(x)
        x = self.conv2(x)
        x = residual1 + self.drop_path(x)
        return x

class FasterNetBlock2_2(nn.Module):
    def __init__(self, dim, expand_ratio=2, act_layer=nn.ReLU, drop_path_rate=0.0):
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
        residual1 = x +residual
        x = self.pconv2(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.act_layer(x)
        x = self.conv2(x)
        x =   residual1 +self.drop_path(x)
        return x




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



# def proposed(dataset, patch_size):
#     model = None
#     if dataset == 'sa':
#         model = MyTransformer(img_size=patch_size, in_chans=204, num_classes=16, n_groups=[16, 16, 16], depths=[2, 1, 1])
#     elif dataset == 'pu':
#         model = MyTransformer(img_size=patch_size, in_chans=103, num_classes=9, n_groups=[2, 2, 2], depths=[1, 2, 1])
#     elif dataset == 'whulk':
#         model = MyTransformer(img_size=patch_size, in_chans=270, num_classes=9, n_groups=[2, 2, 2], depths=[2, 2, 1])
#     elif dataset == 'hrl':
#         model = MyTransformer(img_size=patch_size, in_chans=176, num_classes=14, n_groups=[4, 4, 4], depths=[1, 2, 1])
#     return model

# if __name__ == "__main__":
#     t = torch.randn(size=(3, 1, 204, 7, 7))
#     net = proposed(dataset='sa', patch_size=7)
#     print("output shape:", net(t).shape)
#     model = MyTransformer()
#     model = model.eval()
#     print (model)
#

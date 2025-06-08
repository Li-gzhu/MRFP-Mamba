import math
import torch
import torch.nn as nn
#from rasterio.rio.blocks import blocks
from mamba_ssm import Mamba
from models_our.models import ARConv_Block as AR
from models_our.ARConv import ARConv
#from model_hit import HiT
from vit_pytorch.hit import HiT, ConvPermuteMLP
from pvm import MFPVMLayer, PVMLayer

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

        # self.q = nn.Linear(dim, dim, bias=True)
        # self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.ECA_q = ECANet(dim, dim)
        self.ECA_K = ECANet(dim, dim)
        self.ECA_V = ECANet(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
       # print(x.shape)
        """
        input: (B, N, C)
        B = Batch size, N = patch_size * patch_size, C = dimension for attention
        output: (B, N, C)
        """
        B, N, C = x.shape#100,121,256
        q = self.ECA_q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.ECA_K(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.ECA_V(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # k, v = kv[0], kv[1]
        #kv = self.ELA(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        #print(':', k.shape)#torch.Size([2, 100, 8, 121, 32])


        #print(':', k.shape)#([100, 8, 121, 32])

       # print(k.shape)

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


class Groupe_AR(nn.Module):
    def __init__(self, in_chans=3, embed_dim=128, n_groups=1, num=11):
        super().__init__()
        #self.ifm_size = in_feature_map_size
        #self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1, groups=n_groups)
        #self.proj = ARConv(in_chans, embed_dim, 3, 1, 1)
        self.batch_norm = nn.BatchNorm2d(embed_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):#, e, h
        """
        input: (B, in_chans, in_feature_map_size, in_feature_map_size)
        output: (B, (after_feature_map_size x after_feature_map_size-2), embed_dim = C)
        """
        # print("...before proj shape{}   ".format(x.shape))

        x = self.proj(x)#, e, h
        x = self.relu(self.batch_norm(x))
        # print("...after proj shape{}   ".format(x.shape))

        # x = x.flatten(2).transpose(1, 2)
        # # print("...after flatten shape{}   ".format(x.shape))
        #
        # after_feature_map_size = self.ifm_size

        return x#, after_feature_map_size
class Groupe_MFPVM(nn.Module):
    def __init__(self, in_chans=3, embed_dim=128, n_groups=1, num=11):
        super().__init__()
        #self.ifm_size = in_feature_map_size
        #self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1, groups=n_groups)
        #self.proj = ARConv(in_chans, embed_dim, 3, 1, 1)
        self.batch_norm = nn.BatchNorm2d(embed_dim//4)
        self.relu = nn.ReLU(inplace=True)
        self.batch_norm1 = nn.BatchNorm2d(embed_dim // 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.batch_norm2 = nn.BatchNorm2d(embed_dim // 4)
        self.relu2 = nn.ReLU(inplace=True)
        self.batch_norm3 = nn.BatchNorm2d(embed_dim // 4)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv1x1 = nn.Conv2d(in_chans, embed_dim//4,1,padding=0)
        self.conv3x3 = nn.Conv2d(in_chans, embed_dim//4, 3, padding=1)
        self.conv5x5 = nn.Conv2d(in_chans, embed_dim//4, 5, padding=2)
        self.conv7x7 = nn.Conv2d(in_chans, embed_dim//4, 7, padding=3)

    def forward(self, x):#, e, h
        """
        input: (B, in_chans, in_feature_map_size, in_feature_map_size)
        output: (B, (after_feature_map_size x after_feature_map_size-2), embed_dim = C)
        """
        # print("...before proj shape{}   ".format(x.shape))

        #x1 = self.proj(x)#, e, h
        x1 = self.conv1x1(x)
        x1 = self.relu(self.batch_norm(x1))

        x2 = self.conv3x3(x)
        x2 = self.relu1(self.batch_norm1(x2))

        x3 = self.conv5x5(x)
        x3 = self.relu2(self.batch_norm2(x3))

        x4 = self.conv7x7(x)
        x4 = self.relu3(self.batch_norm3(x4))
        #x = torch.cat([x1, x2, x3, x4], dim=1)
        # print("...after proj shape{}   ".format(x.shape))

        # x = x.flatten(2).transpose(1, 2)
        # # print("...after flatten shape{}   ".format(x.shape))
        #
        # after_feature_map_size = self.ifm_size

        return  x1, x2, x3, x4#, after_feature_map_size
class Groupe_MFPVM2(nn.Module):
    def __init__(self, in_chans=3, embed_dim=128, n_groups=1, num=11):
        super().__init__()
        #self.ifm_size = in_feature_map_size
        #self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1, groups=n_groups)
        #self.proj = ARConv(in_chans, embed_dim, 3, 1, 1)
        self.batch_norm = nn.BatchNorm2d(embed_dim//4)
        self.relu = nn.ReLU(inplace=True)
        self.batch_norm1 = nn.BatchNorm2d(embed_dim // 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.batch_norm2 = nn.BatchNorm2d(embed_dim // 4)
        self.relu2 = nn.ReLU(inplace=True)
        self.batch_norm3 = nn.BatchNorm2d(embed_dim // 4)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv1x1 = nn.Conv2d(in_chans, embed_dim//4,1,padding=0)
        self.conv3x3 = nn.Conv2d(in_chans, embed_dim//4, 3, padding=1)
        self.conv5x5 = nn.Conv2d(in_chans, embed_dim//4, 5, padding=2)
        self.conv7x7 = nn.Conv2d(in_chans, embed_dim//4, 7, padding=3)

    def forward(self, x):#, e, h
        """
        input: (B, in_chans, in_feature_map_size, in_feature_map_size)
        output: (B, (after_feature_map_size x after_feature_map_size-2), embed_dim = C)
        """
        # print("...before proj shape{}   ".format(x.shape))

        #x1 = self.proj(x)#, e, h
        x1 = self.conv1x1(x)
        x1 = self.relu(self.batch_norm(x1))

        x2 = self.conv3x3(x)
        x2 = self.relu1(self.batch_norm1(x2))

        x3 = self.conv5x5(x)
        x3 = self.relu2(self.batch_norm2(x3))

        x4 = self.conv7x7(x)
        x4 = self.relu3(self.batch_norm3(x4))
        x = torch.cat([x1, x2, x3, x4], dim=1)
        # print("...after proj shape{}   ".format(x.shape))

        # x = x.flatten(2).transpose(1, 2)
        # # print("...after flatten shape{}   ".format(x.shape))
        #
        # after_feature_map_size = self.ifm_size

        return x#x1, x2, x3, x4#, after_feature_map_size


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
                n_groups=[16, 16, 16], embed_dims=[256, 128, 64], num_heads=[8, 4, 2], mlp_ratios=[1, 1, 1], depths=[2, 1, 1]):
        super().__init__()

        self.num_stages = num_stages
        
        new_bands = math.ceil(in_chans / n_groups[0]) * n_groups[0]
        self.pad = nn.ReplicationPad3d((0, 0, 0, 0, 0, new_bands - in_chans))

        for i in range(num_stages):
            patch_embed = GroupedPixelEmbedding(
                in_feature_map_size=img_size,
                in_chans=new_bands if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i],
                n_groups=n_groups[i]
            )

            block = nn.ModuleList([Block(
                dim=embed_dims[i], 
                num_heads=num_heads[i],
                mlp_ratio=mlp_ratios[i], 
                drop=0., 
                attn_drop=0.) for j in range(depths[i])])
            
            norm = nn.LayerNorm(embed_dims[i])

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)
        
        self.head = nn.Linear(embed_dims[-1], num_classes)  # 只有pvt时的Head

    def forward_features(self, x):
        # (bs, 1, n_bands, patch size (ps, of HSI), ps)
        # torch.Size([100, 1, 144, 11, 11])#GRSS
        # torch.Size([100, 144, 11, 11])#GRSS
        #print(x.shape)#100, 1, 103, 11, 11 pu
        x = self.pad(x).squeeze(dim=1)
        #print(x.shape)#100, 104, 11, 11 pu
        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            
            x, s = patch_embed(x)  # s = feature map size after patch embedding
            for blk in block:
                x = blk(x)
            
            x = norm(x)
            
            if i != self.num_stages - 1:
                # print("x",x.shape)
                x = x.reshape(B, s, s, -1).permute(0, 3, 1, 2).contiguous()
                #print(x.shape)
        
        return x
    
    def forward(self, x):
        #print("....x.shape{}".format(x.shape))
        x = self.forward_features(x)
        #print("....forward_features   x.shape{}".format(x.shape))

        x = x.mean(dim=1)
        #print(x.shape)
        x = self.head(x)
        #print(x.shape)
        return x


class MRFP_Mamba(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, num_stages=3,
                 embed_dims=[256, 128, 64],
                flatten = True):
        super().__init__()

        self.num_stages = num_stages

        new_bands = math.ceil(in_chans)#(in_chans / n_groups[0]) * n_groups[0]
        self.pad = nn.ReplicationPad3d((0, 0, 0, 0, 0, new_bands - in_chans))

        for i in range(num_stages):
            patch_embed = Groupe_MFPVM(
                in_chans=new_bands if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i],
            )
            block = nn.ModuleList([MFPVMLayer(embed_dims[i], embed_dims[i])])


            # block = nn.ModuleList([Block(
            #     dim=embed_dims[i],
            #     num_heads=num_heads[i],
            #     mlp_ratio=mlp_ratios[i],
            #     drop=0.,
            #     attn_drop=0.) for j in range(depths[i])])

            norm = nn.LayerNorm(embed_dims[i])

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)
        self.is_flatten = flatten
        #self.flatten = nn.Flatten(-2, -1)
        self.head = nn.Linear(embed_dims[-1], num_classes)  # 只有pvt时的Head


    def forward_features(self, x):#
        # (bs, 1, n_bands, patch size (ps, of HSI), ps)
        # torch.Size([100, 1, 144, 11, 11])#GRSS
        # torch.Size([100, 144, 11, 11])#GRSS
        # print(x.shape)#100, 1, 103, 11, 11 pu
        x = self.pad(x).squeeze(dim=1)
        # print(x.shape)#100, 104, 11, 11 pu
        #B = x.shape[0]


        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")

            x, x1, x2, x3 = patch_embed(x)  # s = feature map size after patch embedding , e, [1, 12]
            for blk in block:
            #x = torch.cat([x, x1], dim=1)
                x = blk(x, x1, x2, x3)#
            x = x.permute(0, 3, 2, 1)
            x = norm(x)
            x = x.permute(0, 3, 2, 1)
        # if self.epoch <=100:
        #     self.epoch = self.epoch  + 1
        # else:
        #     self.epoch =0
        # print(':e=',self.epoch)

            # if i != self.num_stages - 1:
            #     # print("x",x.shape)
            #     x = x.reshape(B, s, s, -1).permute(0, 3, 1, 2).contiguous()

        return x

    def forward(self, x):#,e
        #print("....x.shape{}".format(x.shape))
        x = self.forward_features(x)#,e
        #print("....forward_features   x.shape{}".format(x.shape))[100, 64, 15, 15]

        x = x.mean(dim=-1)
        #if (self.is_flatten): x = self.flatten(x)
        #print(x.shape)#100,6400   100, 64, 225
        x = x.mean(dim=-1)
        #print('x1:', x.shape)100, 64
        x = self.head(x)
        return x


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
# if __name__ == '__main__':
#     img_size = 11
#     x = torch.rand(2, 1, 200, img_size, img_size).cuda()
#     net = ARSNet2(num_classes=16,in_chans=200,img_size=11).cuda()
#     # print(net)
#     net.eval()
#
#
#     from thop import profile
#     flops, params = profile(net, (x,))
#
#     print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
#     print('Params = ' + str(params / 1000 ** 2) + 'M')
from vit_pytorch.vit import ViT

#     img_size = 15
#     x = torch.rand(2, 1, 200, img_size, img_size).cuda()
#     #kwargs.setdefault("patch_size", 15)
#     # center_pixel = True
#     # layers = [4, 3, 14, 3]
#     # transitions = [False, True, False, False]
#     # segment_dim = [8, 8, 4, 4]
#     # mlp_ratios = [3, 3, 3, 3]
#     # embed_dims = [480, 480, 256,
#     #               256]  ## for IN 400, for GRSS 256, for PU 168, for KSC 352 for XA 480 for Houston2013 288
#     # net = HiT(layers, img_size=15, in_chans=144, num_classes=16, embed_dims=embed_dims, patch_size=3,
#     #             transitions=transitions,
#     #             segment_dim=segment_dim, mlp_ratios=mlp_ratios, mlp_fn=ConvPermuteMLP, ).cuda()
#     net  = ViT(dim=128, image_size=15, patch_size=3, depth=6, heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1, num_classes=16, channels=200,).cuda()
#     #net = ARSNet2(num_classes=16,in_chans=200,img_size=11).cuda()
#     # print(net)
#     net.eval()
class ARSNet3(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000, num_stages=3,
                 n_groups=[16, 16, 16], embed_dims=[256, 128, 64], num_heads=[8, 4, 2], mlp_ratios=[1, 1, 1],
                flatten = True):
        super().__init__()

        self.num_stages = num_stages

        new_bands = math.ceil(in_chans / n_groups[0]) * n_groups[0]
        self.pad = nn.ReplicationPad3d((0, 0, 0, 0, 0, new_bands - in_chans))

        for i in range(num_stages):
            patch_embed = Groupe_MFPVM2(
                in_chans=new_bands if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i],

                n_groups=n_groups[i],
                num= 11
            )
            block = nn.ModuleList([PVMLayer(embed_dims[i], embed_dims[i])])


            # block = nn.ModuleList([Block(
            #     dim=embed_dims[i],
            #     num_heads=num_heads[i],
            #     mlp_ratio=mlp_ratios[i],
            #     drop=0.,
            #     attn_drop=0.) for j in range(depths[i])])

            norm = nn.LayerNorm(embed_dims[i])

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)
        self.is_flatten = flatten
        self.flatten = nn.Flatten(-2, -1)
        self.head = nn.Linear(embed_dims[-1], num_classes)  # 只有pvt时的Head
        self.epoch = 600

    def forward_features(self, x):#
        # (bs, 1, n_bands, patch size (ps, of HSI), ps)
        # torch.Size([100, 1, 144, 11, 11])#GRSS
        # torch.Size([100, 144, 11, 11])#GRSS
        # print(x.shape)#100, 1, 103, 11, 11 pu
        x = self.pad(x).squeeze(dim=1)
        # print(x.shape)#100, 104, 11, 11 pu
        #B = x.shape[0]


        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")

            x = patch_embed(x)  # s = feature map size after patch embedding , e, [1, 12]
            for blk in block:
                x = blk(x)#
            x = x.permute(0, 3, 2, 1)
            x = norm(x)
            x = x.permute(0, 3, 2, 1)
        # if self.epoch <=100:
        #     self.epoch = self.epoch  + 1
        # else:
        #     self.epoch =0
        # print(':e=',self.epoch)

            # if i != self.num_stages - 1:
            #     # print("x",x.shape)
            #     x = x.reshape(B, s, s, -1).permute(0, 3, 1, 2).contiguous()

        return x

    def forward(self, x):#,e
        #print("....x.shape{}".format(x.shape))
        x = self.forward_features(x)#,e
        # print("....forward_features   x.shape{}".format(x.shape))

        #x = x.mean(dim=1)
        if (self.is_flatten): x = self.flatten(x)
        #print(x.shape)#100,6400
        x = x.mean(dim=-1)
        x = self.head(x)
        return x
#
#     from thop import profile
#     flops, params = profile(net, (x,))
#
#     print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
#     print('Params = ' + str(params / 1000 ** 2) + 'M')
from thop import profile, clever_format
from torchinfo import summary
from vit_pytorch.hit import HiT, ConvPermuteMLP
from ssftt import SSFTT
from SSFTTnet import SSFTTnet
from conv2d import TESTEtAl
from conv3d import C3DEtAl
from vit_pytorch.pit import PiT
#img_size = 15
#     x = torch.rand(2, 1, 200, img_size, img_size).cuda()
if __name__ == '__main__':
    input_tensor = torch.randn(2, 1, 103, 15, 15)
    input_tensor = input_tensor.cuda()
    model = model = PiT(dim=256, image_size=15, patch_size=3, depth=(3, 3, 3, 3), heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1, num_classes=9, channels=103,).cuda()#ViT(dim=128, image_size=15, patch_size=3, depth=6, heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1, num_classes=16, channels=103,).cuda()
   #SSFTTnet(num_classes=9).cuda()#gaht(img_size=15,#(img_size=patch_size)n_groups=[2,2,2], depths=[1,1,1], num_classes=9, in_chans=103,).cuda()#ARSNet2(num_classes=9, in_chans=103,).cuda()#ViT(dim=128, image_size=15, patch_size=3, depth=6, heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1,
                #num_classes=9, channels=103,).cuda()#
        # (
        # gaht(img_size=15,#(img_size=patch_size)
        #     n_groups=[2,2,2], depths=[1,1,1], num_classes=9, in_chans=103,
        #                   ).cuda())#ARSNet2(img_size=15, n_groups=[4,4,4], num_classes=9, in_chans=103,).cuda()#ViT(dim=128, image_size=15, patch_size=3, depth=6, heads=16, mlp_dim=2048, dropout=0.1, emb_dropout=0.1,
                #num_classes=9, channels=103,).cuda()

    # layers = [4, 3, 14, 3]
    # transitions = [False, True, False, False]
    # segment_dim = [8, 8, 4, 4]
    # mlp_ratios = [3, 3, 3, 3]
    # embed_dims = [320, 320, 256,
    #               256]  ## for IN 400, for GRSS 256, for PU 168, for KSC 352 for XA 480 for Houston2013 288
    # model = HiT(layers, img_size=15, in_chans=103, num_classes=9, embed_dims=embed_dims, patch_size=3,
    #             transitions=transitions,
    #             segment_dim=segment_dim, mlp_ratios=mlp_ratios, mlp_fn=ConvPermuteMLP, ).cuda()
    # model  = ARSNet2(img_size=15,#(img_size=patch_size)
    #         n_groups=[4,4,4], num_classes=16, in_chans=144,
    #                       ).cuda() #
#

# 计算参数量
    summary_info = summary(model, input_size=(2, 1, 103, 15, 15))
    print("\nModel Summary (Parameters in M):")
    print(f"Total parameters: {summary_info.total_params / 1e6:.2f}M")

     # 计算计算量
    macs, params = profile(model, inputs=(input_tensor,))#
    macs, params = clever_format([macs, params], "%.2f")

    print("\nModel Computational Cost (in GFLOPs):")
    print(f"MACs: {macs}, Parameters: {params}")
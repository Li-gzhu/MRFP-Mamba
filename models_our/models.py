import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from .ARConv import ARConv
from mamba_ssm import Mamba
from pvm import PVMLayer as pvm

class ARConv_Block(nn.Module):
    def __init__(self, in_planes, flag=False):
        super(ARConv_Block, self).__init__()
        self.flag = flag
        # self.conv1 = ARConv(in_planes, in_planes, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.in_planes = in_planes
        self.norm = nn.LayerNorm(in_planes)
        self.norm1 = nn.LayerNorm(in_planes)
        #self.conv2 = ARConv(in_planes, in_planes, 3, 1, 1)
        # self.mamba = Mamba(d_model=in_planes, d_state=16, d_conv=4, expand=2)
        self.mamba = pvm(input_dim=in_planes,output_dim=in_planes, d_state=16, d_conv=4, expand=2)
        self.skip_scale = nn.Parameter(torch.ones(1))
        #self.proj = nn.Linear()

    def forward(self, x):#, epoch, hw_range
        x = self.mamba(x)
        # res = self.conv1(x, epoch, hw_range)
        # res = self.relu(res)
        # #res = self.norm1(res)
        # #res = self.conv2(res, epoch, hw_range)
        # x = x + res
        #x1 =x
        # if x1.dtype == torch.float16:
        #     x1 = x1.type(torch.float32)
        # B, C = x1.shape[:2]
        # assert C == self.in_planes
        # n_tokens = x1.shape[2:].numel()
        # img_dims = x.shape[2:]
        # x_flat1 = x1.reshape(B, C, n_tokens).transpose(-1, -2)
        # x_norm1 = self.norm(x_flat1)
        # x_mamba = self.mamba(x_norm1) + self.skip_scale * x_norm1
        # x_mamba = self.norm(x_mamba)
        # x_mamba = self.relu(x_mamba)
        # x = x_mamba.transpose(-1, -2).reshape(B, self.in_planes, *img_dims)

        return x
class ARSNet(nn.Module):
    def __init__(self, input_channels, num_classes, patch_size=15, flatten = True):
        super().__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.aux_loss_weight = 1
        #self.pad = nn.ReplicationPad3d((0, 0, 0, 0, 0, new_bands - in_chans))
        self.inc = Inc(input_channels)#nn.Conv2d(input_channels, 64, 3,1)#Inc(input_channels)
        #self.inc = ConvDown(input_channels, 256)
        self.rb1 = ARConv_Block(256, flag=True)
        self.down1 = ConvDown(256, 128)
        #self.down1 = ConvDown1(64)
        self.rb2 = ARConv_Block(128)
        #self.down2 = ConvDown1(128)
        self.down2 = ConvDown(128, 64)
        self.rb3 = ARConv_Block(64)
        self.head = nn.Linear(64, num_classes)
        self.is_flatten = flatten
        self.flatten = nn.Flatten(-2, -1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, x, epoch):
        #print(x.shape)#100,1,103,11,11

        x = x.squeeze()
        x = self.inc(x)
        x = self.rb1(x, epoch, [0,18])
        x = self.down1(x)
        x = self.rb2(x, epoch, [0,18])
        x = self.down2(x)
        x = self.rb3(x, epoch, [0,18])
        #print(x.shape)#100,256,5,5
        x = self.pool(x)
        x = x.squeeze(dim=-1).squeeze(dim=-1)
        #if (self.is_flatten): x = self.flatten(x)
        #print(x.shape)#100,6400

        #x = x.mean(dim=-1)
        # #print(x.shape)
        # x = x.mean(dim=-1)
        #print(x.shape)#100,256,5
        #x = x.mean(dim=1)
        x = self.head(x)
        #print(x.shape)
        return x

class ARNet(nn.Module):
    def __init__(self, pan_channels=1, lms_channels=8):
        super(ARNet, self).__init__()
        self.head_conv = nn.Conv2d(pan_channels + lms_channels, 32, 3, 1, 1)
        self.rb1 = ARConv_Block(32, flag=True)
        self.down1 = ConvDown(32)
        self.rb2 = ARConv_Block(64)
        self.down2 = ConvDown(64)
        self.rb3 = ARConv_Block(128)
        self.up1 = ConvUp(128)
        self.rb4 = ARConv_Block(64)
        self.up2 = ConvUp(64)
        self.rb5 = ARConv_Block(32)
        self.tail_conv = nn.Conv2d(32, lms_channels, 3, 1, 1)
 
    def forward(self, pan, lms, epoch, hw_range):
        x1 = torch.cat([pan, lms], dim=1)
        x1 = self.head_conv(x1)
        x1 = self.rb1(x1, epoch, hw_range)
        x2 = self.down1(x1)
        x2 = self.rb2(x2, epoch, hw_range)
        x3 = self.down2(x2)
        x3 = self.rb3(x3, epoch, hw_range)
        x4 = self.up1(x3, x2)
        del x2
        x4 = self.rb4(x4, epoch, hw_range)
        x5 = self.up2(x4, x1)
        del x1
        x5 = self.rb5(x5, epoch, hw_range)
        x5 = self.tail_conv(x5)
        return lms + x5

# Downsample and Upsample blocks for Unet-strucutre

class ConvDown(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # if dsconv:
        #     self.conv = nn.Sequential(
        # self.proj = nn.Conv2d(in_channels, out_channels, 3, 1,1, groups=2),
        # self.bn = nn.BatchNorm2d(out_channels),
        # self.relu = nn.ReLU(inplace=True)
        self.proj = nn.ConvTranspose2d(in_channels, out_channels, 2, 2, 0)#nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=2)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
                # nn.LeakyReLU(inplace=True),
                # nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False),
                # nn.Conv2d(in_channels, in_channels * 2, 1, 1, 0)
            #)
        # else:
        #     self.conv = nn.Sequential(
        #         nn.Conv2d(in_channels, in_channels, 3, 2, 1),
        #         nn.LeakyReLU(inplace=True),
        #         nn.Conv2d(in_channels, in_channels * 2, 3, 1, 1)
        #     )

    def forward(self, x):
        # return self.conv(x)
        x = self.proj(x)
        x = self.relu(self.batch_norm(x))
        return x
class ConvDown1(nn.Module):
    def __init__(self, in_channels, dsconv=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if dsconv:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 2, 2, 0),
                # nn.BatchNorm2d(in_channels),
                # nn.ReLU(inplace=True)
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False),
                nn.Conv2d(in_channels, in_channels * 2, 1, 1, 0)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, 2, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels * 2, 3, 1, 1)
            )

    def forward(self, x):
        return self.conv(x)


class Inc(nn.Module):
    def __init__(self, in_channels, dsconv=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        if dsconv:
            self.conv = nn.Sequential(
                # nn.Conv2d(in_channels, in_channels, 2, 2, 0),
                # nn.LeakyReLU(inplace=True),
                # nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False),
                nn.Conv2d(in_channels, 256, 3, 1, 1)
            )
            self.batch_norm = nn.BatchNorm2d(256)
            self.relu = nn.ReLU(inplace=True)
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, 2, 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(in_channels, 256, 3, 1, 1)
            )

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(self.batch_norm(x))
        return x
 
 
class ConvUp(nn.Module):
    def __init__(self, in_channels, dsconv=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
 
        self.conv1 = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, 2, 0)
        if dsconv:
            self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels // 2, in_channels // 2, 3, 1, 1, groups=in_channels // 2, bias=False),
                nn.Conv2d(in_channels // 2, in_channels // 2, 1, 1, 0)
            )
        else:
            self.conv2 = nn.Conv2d(in_channels // 2, in_channels // 2, 3, 1, 1)
 
    def forward(self, x, y):
        x = F.leaky_relu(self.conv1(x))
        x = x + y
        x = F.leaky_relu(self.conv2(x))
        return x
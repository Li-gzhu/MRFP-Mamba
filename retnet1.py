import torch
import torch.nn as nn
from einops import rearrange
from retention import MultiScaleRetention



NUM_CLASS = 16

class Convnet(nn.Module):
    def __init__(self, in_channels=1):
        super(Convnet, self).__init__()
        # self.L = num_tokens
        # self.cT = dim
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=8, kernel_size=(3, 3, 3)),#通过卷积运算生成8个11 × 11 × 28的特征立方体
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )#x = rearrange(x, 'b c h w y -> b (c h) w y')
#平面化成一个1-D特征向量，得到大小为1×81的64个向量
        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=8*28, out_channels=64, kernel_size=(3, 3)),#将八个特征立方体重新排列，生成一个11 × 11 × 224的特征立方体。作为2d卷积的输入。得到64*9*9；
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )


        # self.conv2d_features2 = nn.Sequential(
        #     nn.Conv2d(in_channels=112, out_channels=64, kernel_size=(2, 2)),#得到64*8*8；
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        # )

    def forward(self, X, mask=None):
        X = self.conv3d_features(X)  # patch is 13×13×30,8颗3×3×3立方核,通过卷积运算生成8个11 × 11 × 28的特征立方体
        X = rearrange(X, 'b c h w y -> b (c h) w y')
        X = self.conv2d_features(X)  # 64个3 × 3平面核，得到64个大小为9×9。
       # X = self.conv2d_features2(X)  #64*8*8
        X = rearrange(X, 'b c h w -> b (h w) c') #64,81,64
        return X







class RetNet(nn.Module):
    def __init__(self, layers=2, hidden_dim=64, ffn_size=128, heads=4, dim1=5184, dim2=None, num_classes=NUM_CLASS,double_v_dim=False):
        super(RetNet, self).__init__()
        #添加
        self.num_classes = NUM_CLASS
        self.convnet = Convnet(in_channels=1)


        #
        self.list_retnet={"layers":layers,"hidden_dim":hidden_dim,"ffn_size":ffn_size,"heads":heads,"dim1":dim1,"dim2":dim2,"num_classes":num_classes}

        self.layers = layers
        self.hidden_dim = hidden_dim
        self.ffn_size = ffn_size
        self.heads = heads
        self.v_dim = hidden_dim * 2 if double_v_dim else hidden_dim

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

        self.nn1 = nn.Linear(dim1, num_classes)  #
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bias, std=1e-6)
        self.flatten = nn.Flatten(start_dim=1)

        # self.nn2 = nn.Linear(dim2, num_classes)
        # torch.nn.init.xavier_uniform_(self.nn2.weight)
        # torch.nn.init.normal_(self.nn2.bias, std=1e-6)





    def forward(self, X):
        """
        X: (batch_size, sequence_length, hidden_size)
        """
        X = self.convnet(X)  #  X(64,64,64)  64个8*8
       # X = torch.transpose(X, -1, -2)  #X(64,64,64)
        for i in range(self.layers):
            Y = self.retentions[i](self.layer_norms_1[i](X)) + X  #  Y(64,81,64)
           
            X = self.ffns[i](self.layer_norms_2[i](Y)) + Y  #  X(64,81,64)
            if i == (self.layers-1) :
                X = self.flatten(X)
                X = self.nn1(X)
               # X = self.nn2(X)
        out=self.list_retnet
        return X,out

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





if __name__ == '__main__':
    model = RetNet()
    model.eval()
    print(model)
    input = torch.randn(64, 1, 30, 13, 13)
    y = model(input)
    print(y.size())
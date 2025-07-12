import torch
import torch.nn as nn
import torch.nn.functional as F

class NONLocalBlock2D(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        """
        Non-local Block for 2D feature maps
        :param in_channels: 輸入通道數
        :param inter_channels: 中間通道數（預設為 in_channels // 2）
        :param sub_sample: 是否對 phi 和 g 做下采樣，節省計算
        :param bn_layer: 是否在最後加 BatchNorm
        """
        super(NONLocalBlock2D, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels if inter_channels is not None else in_channels // 2
        if self.inter_channels == 0:
            self.inter_channels = 1

        # θ, φ, g 都是 1x1 卷積
        self.g = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.theta = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)

        # W_z 用來恢復通道數 + BatchNorm
        if bn_layer:
            self.W_z = nn.Sequential(
                nn.Conv2d(self.inter_channels, in_channels, kernel_size=1),
                nn.BatchNorm2d(in_channels)
            )
            # 初始化 BN，讓初始狀態近似於恒等映射
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1)
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        # 是否下采樣
        if sub_sample:
            self.g = nn.Sequential(self.g, nn.MaxPool2d(kernel_size=2))
            self.phi = nn.Sequential(self.phi, nn.MaxPool2d(kernel_size=2))

    def forward(self, x):
        """
        :param x: 輸入特徵圖 (batch, C, H, W)
        :return: 輸出特徵圖 (batch, C, H, W)
        """
        batch_size = x.size(0)

        # θ 分支 (B, C', H*W)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)  # (B, H*W, C')

        # φ 分支 (B, C', H'*W')
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)  # (B, C', H'*W')

        # 內積得到注意力圖 (B, H*W, H'*W')
        f = torch.matmul(theta_x, phi_x)

        # softmax 正規化
        f_div_C = F.softmax(f, dim=-1)

        # g 分支 (B, C', H'*W')
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)  # (B, H'*W', C')

        # 注意力圖加權 (B, H*W, C')
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()  # (B, C', H*W)
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])  # (B, C', H, W)

        # 恢復通道數 + 殘差連接
        W_y = self.W_z(y)
        z = W_y + x

        return z

class ResnetBlock(nn.Module):
    """ResNet block with configurable normalization, dropout, and padding."""

    def __init__(self, dim, padding_type='reflect', norm_layer=nn.InstanceNorm2d, use_dropout=False, use_bias=True):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
        self.non_local = NONLocalBlock2D(dim, sub_sample=False, bn_layer=True)


    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0

        # Padding type
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(f'Padding type [{padding_type}] is not supported')

        # First conv
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True)
        ]

        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        # Padding again
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1

        # Second conv
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim)
        ]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = self.conv_block(x)
        out = self.non_local(out)
        return x + out

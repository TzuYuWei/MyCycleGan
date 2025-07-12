import torch
import torch.nn as nn
import torch.nn.functional as F

class PAM_Module(nn.Module):
    """ Position Attention Module 空間注意力模塊 """
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.channel_in = in_dim

        # 1x1 conv 降維
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        # 可學參數，初始為 0
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        x: 輸入特徵圖 [B, C, H, W]
        """
        B, C, H, W = x.size()
        # [B, C/8, H*W]
        proj_query = self.query_conv(x).view(B, -1, H*W).permute(0, 2, 1)  # [B, H*W, C/8]
        proj_key = self.key_conv(x).view(B, -1, H*W)                        # [B, C/8, H*W]
        energy = torch.bmm(proj_query, proj_key)                            # [B, H*W, H*W]
        attention = self.softmax(energy)

        proj_value = self.value_conv(x).view(B, -1, H*W)                    # [B, C, H*W]

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))             # [B, C, H*W]
        out = out.view(B, C, H, W)

        out = self.gamma * out + x
        return out


class CAM_Module(nn.Module):
    """ Channel Attention Module 通道注意力模塊 """
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.channel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        x: 輸入特徵圖 [B, C, H, W]
        """
        B, C, H, W = x.size()
        # [B, C, H*W]
        proj_query = x.view(B, C, -1)
        proj_key = x.view(B, C, -1).permute(0, 2, 1)        # [B, H*W, C]
        energy = torch.bmm(proj_query, proj_key)            # [B, C, C]

        # trick: 最大值減去 energy
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)                # [B, C, C]

        proj_value = x.view(B, C, -1)

        out = torch.bmm(attention, proj_value)              # [B, C, H*W]
        out = out.view(B, C, H, W)

        out = self.gamma * out + x
        return out

class ResnetBlock(nn.Module):
    """ResNet block with configurable normalization, dropout, and padding."""

    def __init__(self, dim, padding_type='reflect', norm_layer=nn.InstanceNorm2d, use_dropout=False, use_bias=True):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
        self.pam = PAM_Module(dim)
        self.cam = CAM_Module(dim)


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
        out = out + self.pam(out) + self.cam(out)
        return x + out


import torch
import torch.nn as nn
import time
from thop import profile

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === 原版 ResnetBlock ===
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.InstanceNorm2d, use_dropout=False, use_bias=True):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True)
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim)
        ]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = self.conv_block(x)
        return x + out

# === SE Layer ===
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# === 加 SE 的 ResnetBlock ===
class ResnetBlockSE(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.InstanceNorm2d, use_dropout=False, use_bias=True):
        super(ResnetBlockSE, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
        self.se_block = SELayer(dim)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        # 同上
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1

        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim),
            nn.ReLU(True)
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        conv_block += [
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
            norm_layer(dim)
        ]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = self.conv_block(x)
        out = self.se_block(out)
        return x + out

# === Generator（重點是中間有 9 個 ResnetBlock）===
class Generator(nn.Module):
    def __init__(self, resblock, input_nc=3, output_nc=3, ngf=64, n_blocks=9):
        super(Generator, self).__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf*4),
            nn.ReLU(inplace=True),
        ]
        for _ in range(n_blocks):
            model += [resblock(dim=ngf*4)]
        model += [
            nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf*2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf*2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# === 測試函式 ===
def test_speed(model, name):
    model.to(device)
    model.eval()
    x = torch.randn(8, 3, 256, 256).to(device)
    # 預熱
    for _ in range(10):
        _ = model(x)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        _ = model(x)
    torch.cuda.synchronize()
    avg_time = (time.time() - start) / 10
    print(f"{name}: 平均每次 forward {avg_time:.4f} 秒")
    # 計算 FLOPs 和參數量
    macs, params = profile(model, inputs=(x,), verbose=False)
    print(f"{name}: Params={params/1e6:.2f}M, FLOPs={macs/1e9:.2f}G")

if __name__ == "__main__":
    print("開始測試...")
    model_plain = Generator(resblock=ResnetBlock)
    model_se = Generator(resblock=ResnetBlockSE)
    test_speed(model_plain, "原版Generator")
    test_speed(model_se, "加SE Generator")

# import PyTorch pacakges
import csv
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19, VGG19_Weights
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from PIL import Image
from torchmetrics import JaccardIndex
from thop import profile
import os
import itertools
import torch.fft
import math
import time
import cv2
import numpy as np
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torchvision import transforms
from piq import ssim, psnr, FID, LPIPS  # 使用 piq 套件中的 LPIPS 和 FID
import lpips
from torchvision.models import inception_v3
import re
import matplotlib.pyplot as plt
import random
import shutil
from torchvision.transforms import InterpolationMode
import collections
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device)

# 你想要儲存的資料夾路徑
TXT_dir = r'C:\Users\User\Desktop\CycleGAN_flip_128\result\train_mean'
save_dir = r'C:\Users\User\Desktop\CycleGAN_flip_128\result\test result_128'
model_dir = r'C:\Users\User\Desktop\CycleGAN_flip_128\models'
loss_dir = r'C:\Users\User\Desktop\CycleGAN_flip_128\loss_plot'
loss_csv_path = os.path.join(loss_dir, "train_loss_log.csv")

# 可學習的頻率索引
class LearnableFrequencies(nn.Module):
    def __init__(self, num_freqs, scale_factor, max_width, max_height):
        super().__init__()
        self.freq_u = nn.Parameter(torch.rand(num_freqs) * scale_factor).to(device)
        self.freq_v = nn.Parameter(torch.rand(num_freqs) * scale_factor).to(device)
        self.max_width = max_width
        self.max_height = max_height

    def forward(self):
        # 確保頻率索引在合法範圍內
        fidx_u = (self.freq_u.clamp(0, self.max_width - 1)).int()
        fidx_v = (self.freq_v.clamp(0, self.max_height - 1)).int()
        return fidx_u.tolist(), fidx_v.tolist()

# **Spectral Normalization 添加**
class SpectralNormConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(SpectralNormConv2d, self).__init__()
        self.conv = nn.utils.spectral_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        )

    def forward(self, x):
        return self.conv(x)

class ResnetBlock(nn.Module):
    """ResNet block with configurable normalization, dropout, and padding."""

    def __init__(self, dim, padding_type='reflect', norm_layer=nn.InstanceNorm2d, use_dropout=False, use_bias=True):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

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
        return x + out

# === Generator with RES  ===
class Generator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=6):
        super(Generator, self).__init__()

        # Encoder
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),

            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
        ]

        # Residual blocks 
        for _ in range(n_blocks):
            model += [ResnetBlock(dim=ngf * 4, padding_type='reflect', norm_layer=nn.InstanceNorm2d, use_dropout=False, use_bias=True)]

        # Decoder
        model += [
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# **多尺度 PatchGAN 判別器**
class MultiScalePatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, n_layers=3, num_scales=3):
        super().__init__()
        self.discriminators = nn.ModuleList([
            SpectralPatchGANDiscriminator(in_channels, base_channels, n_layers)
            for _ in range(num_scales)
        ])
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)

    def forward(self, x):
        outputs = []
        for i, d in enumerate(self.discriminators):
            if i > 0:
                x = self.downsample(x)
            outputs.append(d(x))
        return outputs

# **改進版 PatchGAN 判別器**
class SpectralPatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(SpectralPatchGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            SpectralNormConv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNormConv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNormConv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNormConv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNormConv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)

# TVLoss
class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=0.01):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]
   
# **VGG 特徵提取器 (支持多層特徵)**
class VGGFeatureExtractor(nn.Module):
    def __init__(self, layers=[3, 8, 17, 26]):
        super(VGGFeatureExtractor, self).__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features  # 修改這一行
        # 根據層索引生成對應的切片
        self.slices = nn.ModuleList([nn.Sequential(*[vgg[i] for i in range(start, end)])
                                     for start, end in zip([0] + layers[:-1], layers)])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        features = []
        for slice in self.slices:
            x = slice(x)
            features.append(x)
        return features

def frequency_loss(real_img, fake_img, weight=None, eps=1e-8):
    # 計算 FFT
    real_fft = torch.fft.fft2(real_img, norm='ortho')
    fake_fft = torch.fft.fft2(fake_img, norm='ortho')

    # 拆成實部與虛部
    real_real = real_fft.real
    real_imag = real_fft.imag
    fake_real = fake_fft.real
    fake_imag = fake_fft.imag

    # 計算 L1 loss（可以選擇加權 focal mask）
    if weight is not None:
        loss_real = F.l1_loss(real_real * weight, fake_real * weight)
        loss_imag = F.l1_loss(real_imag * weight, fake_imag * weight)
    else:
        loss_real = F.l1_loss(real_real, fake_real)
        loss_imag = F.l1_loss(real_imag, fake_imag)

    loss = loss_real + loss_imag
    return loss

def align_images(*images):
    min_height = min(img.size(2) for img in images)
    min_width = min(img.size(3) for img in images)
    return [img[:, :, :min_height, :min_width] for img in images]

# === Unpaired Dataset（訓練用） ===
class UnpairedImageDataset(Dataset):
    def __init__(self, root_A, root_B, transform=None):
        self.transform = transform

        self.images_A = []
        for city in os.listdir(root_A):
            city_dir = os.path.join(root_A, city)
            self.images_A += [os.path.join(city_dir, f) for f in os.listdir(city_dir) if f.endswith('.png')]

        self.images_B = []
        for city in os.listdir(root_B):
            city_dir = os.path.join(root_B, city)
            self.images_B += [os.path.join(city_dir, f) for f in os.listdir(city_dir) if f.endswith('.png')]

    def __len__(self):
        return max(len(self.images_A), len(self.images_B))

    def __getitem__(self, idx):
        img_A_path = random.choice(self.images_A)
        img_B_path = random.choice(self.images_B)

        img_A = Image.open(img_A_path).convert("RGB")
        img_B = Image.open(img_B_path).convert("RGB")

        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)

        return img_A, img_B

class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images
    

# === 訓練函數（不含 val） ===
def train_cyclegan_unpaired(generator_A2B, generator_B2A, discriminator_A, discriminator_B, dataloader, device, fake_A_pool, fake_B_pool):
    criterion_gan = nn.MSELoss()  # 對抗損失
    criterion_cycle = nn.L1Loss()  # 循環一致性損失
    criterion_perceptual = nn.L1Loss()  # 感知損失
    criterion_identity = nn.L1Loss()  # 身份損失

    vgg = VGGFeatureExtractor().to(device)
    tv_loss = TVLoss(tv_loss_weight=0.01).to(device)  # 初始化 TVLoss

    optimizer_G = optim.Adam(
        itertools.chain(generator_A2B.parameters(), generator_B2A.parameters()),
        lr=0.0002, betas=(0.5, 0.999)
    )
    optimizer_D_A = optim.Adam(discriminator_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(discriminator_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=30, gamma=0.5)
    scheduler_D_A = torch.optim.lr_scheduler.StepLR(optimizer_D_A, step_size=30, gamma=0.5)
    scheduler_D_B = torch.optim.lr_scheduler.StepLR(optimizer_D_B, step_size=30, gamma=0.5)

    start_time = time.time()

    for epoch in range(100):
        for i, (real_A, real_B) in enumerate(dataloader):
            real_A = real_A.to(device)
            real_B = real_B.to(device)

            # Generator Forward
            fake_B = generator_A2B(real_A)
            fake_A = generator_B2A(real_B)
            recon_A = generator_B2A(fake_B)
            recon_B = generator_A2B(fake_A)

            optimizer_G.zero_grad()
            
            # GAN loss
            loss_GAN_A2B = criterion_gan(discriminator_B(fake_B), torch.ones_like(discriminator_B(fake_B)))
            loss_GAN_B2A = criterion_gan(discriminator_A(fake_A), torch.ones_like(discriminator_A(fake_A)))
            
            # Cycle loss
            loss_cycle = criterion_cycle(recon_A, real_A) + criterion_cycle(recon_B, real_B)
            
            # Identity loss
            identity_A = generator_B2A(real_A)
            identity_B = generator_A2B(real_B)
            loss_identity = criterion_identity(identity_A, real_A) + criterion_identity(identity_B, real_B)
            
            # Perceptual loss
            feat_real_A = vgg(real_A)
            feat_fake_B = vgg(fake_B)
            feat_recon_A = vgg(recon_A)
            perceptual_loss = sum([criterion_perceptual(r, f) + criterion_perceptual(r, c)
                                for r, f, c in zip(feat_real_A, feat_fake_B, feat_recon_A)])

            # 頻率損失 + TV 損失
            freq_loss_val = frequency_loss(real_A, fake_B)
            tv_loss_val = tv_loss(fake_B) + tv_loss(fake_A)

            # Total generator loss
            loss_G = (
                loss_GAN_A2B + loss_GAN_B2A +
                10 * loss_cycle + 
                0.5 * loss_identity +
                0.1 * perceptual_loss +
                min(0.1 + epoch * 0.01, 0.5) * freq_loss_val +
                tv_loss_val
            )

            loss_G.backward()
            optimizer_G.step()

            # Update Discriminator A
            optimizer_D_A.zero_grad()
            fake_A_for_D = fake_A_pool.query(fake_A.detach())
            loss_D_A_real = criterion_gan(discriminator_A(real_A), torch.ones_like(discriminator_A(real_A)))
            loss_D_A_fake = criterion_gan(discriminator_A(fake_A_for_D), torch.zeros_like(discriminator_A(real_A)))
            loss_D_A = 0.5 * (loss_D_A_real + loss_D_A_fake)
            loss_D_A.backward()
            optimizer_D_A.step()

            # Update Discriminator B
            optimizer_D_B.zero_grad()
            fake_B_for_D = fake_B_pool.query(fake_B.detach())
            loss_D_B_real = criterion_gan(discriminator_B(real_B), torch.ones_like(discriminator_B(real_B)))
            loss_D_B_fake = criterion_gan(discriminator_B(fake_B_for_D), torch.zeros_like(discriminator_B(real_B)))
            loss_D_B = 0.5 * (loss_D_B_real + loss_D_B_fake)
            loss_D_B.backward()
            optimizer_D_B.step()

        scheduler_G.step()
        scheduler_D_A.step()
        scheduler_D_B.step()

        print(f"Epoch [{epoch+1}/100] | G Loss: {loss_G.item():.4f} | D_A: {loss_D_A.item():.4f} | D_B: {loss_D_B.item():.4f}")
        with open(loss_csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, loss_G.item(), loss_D_A.item(), loss_D_B.item()])

        torch.save({
            'generator_A2B': generator_A2B.state_dict(),
            'generator_B2A': generator_B2A.state_dict(),
            'discriminator_A': discriminator_A.state_dict(),
            'discriminator_B': discriminator_B.state_dict()
        }, os.path.join(model_dir, f"checkpoint_epoch{epoch+1}.pth"))
        elapsed_time = time.time() - start_time  # 計算總時間    
        print(f"✔ 模型已儲存於 checkpoint_epoch{epoch+1}.pth")
        print(f"Epoch [{epoch+1}/100] 訓練時間: {elapsed_time:.2f} 秒")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rain_root = r'C:\Users\User\Desktop\CycleGAN_flip_128\train\trainA'
    sun_root = r'C:\Users\User\Desktop\CycleGAN_flip_128\train\trainB'

    transform = transforms.Compose([
        transforms.Resize((128, 128), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])

    train_dataset = UnpairedImageDataset(rain_root, sun_root, transform=transform)

    # === 🔍 額外檢查資料夾與圖片數 ===
    import collections
    from pathlib import Path

    def count_images_by_city(image_paths):
        counter = collections.defaultdict(int)
        for path in image_paths:
            city = Path(path).parts[-2]  # 倒數第二層資料夾
            counter[city] += 1
        return counter

    print("✔ 總圖片數 A:", len(train_dataset.images_A))
    print("✔ 總圖片數 B:", len(train_dataset.images_B))
    print("✔ 資料集長度（max）:", len(train_dataset))

    print("\n🧾 Rain A 每個城市圖片數：")
    for city, count in count_images_by_city(train_dataset.images_A).items():
        print(f"  - {city}: {count} 張")

    print("\n🧾 Sun B 每個城市圖片數：")
    for city, count in count_images_by_city(train_dataset.images_B).items():
        print(f"  - {city}: {count} 張")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

    generator_A2B = Generator().to(device)
    generator_B2A = Generator().to(device)
    discriminator_A = SpectralPatchGANDiscriminator().to(device)
    discriminator_B = SpectralPatchGANDiscriminator().to(device)

    fake_A_pool = ImagePool(pool_size=50)
    fake_B_pool = ImagePool(pool_size=50)

    # 正式訓練
    train_cyclegan_unpaired(generator_A2B, generator_B2A, discriminator_A, discriminator_B,
                            train_loader, device, fake_A_pool, fake_B_pool)


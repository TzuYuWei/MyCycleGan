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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device)

# 你想要儲存的資料夾路徑
base_dir = r'C:\Users\User\Desktop\小城市測試\model\\'
loss_plot_dir = os.path.join(base_dir, "loss_plot")
log_dir = r"C:\Users\User\Desktop\小城市測試\model\logs"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(loss_plot_dir, exist_ok=True)

# 初始化 IoU 計算器（針對 21 類別 segmentation）
miou_metric = JaccardIndex(task="multiclass", num_classes=21).to(device)

def calculate_miou(pred_mask, gt_mask):
    """
    計算 segmentation mIoU，確保 pred 和 target 形狀符合要求
    """
    if pred_mask.shape[1] == 3:  # 檢查是否是 RGB 影像
        pred_mask = torch.softmax(pred_mask, dim=1)  # 轉換為機率
        pred_mask = torch.argmax(pred_mask, dim=1)  # 取最大值索引作為預測類別

    if gt_mask.shape[1] == 3:  # 若 GT 是 RGB，轉換為單通道
        gt_mask = torch.argmax(gt_mask, dim=1)

    return miou_metric(pred_mask, gt_mask).item()

def compute_flops_params(model, input_shape=(1, 3, 128, 128)):
    """
    計算 FLOPs 和參數量
    """
    dummy_input = torch.randn(input_shape).to(device)
    flops, params = profile(model, inputs=(dummy_input,))
    return flops, params

# 初始化 InceptionV3 模型（作為特徵提取器）
inception = inception_v3(pretrained=True, transform_input=False).to(device)
inception.fc = torch.nn.Identity()  # 去除全連接層，輸出特徵向量
inception.eval()

def extract_features(img, model, device):
    img = F.interpolate(img, size=(299, 299), mode='bilinear', align_corners=False)
    with torch.no_grad():
        features = model(img.to(device))
    return features

# 初始化 LPIPS 和 FID 指標
lpips_fn = lpips.LPIPS(net='vgg').to(device) 

def calculate_metrics(real_img, fake_img):
    real_img, fake_img = align_images(real_img, fake_img)
    real_img = F.interpolate(real_img, size=(299, 299), mode='bilinear', align_corners=False)
    fake_img = F.interpolate(fake_img, size=(299, 299), mode='bilinear', align_corners=False)
    real_img = real_img.to(device)
    fake_img = fake_img.to(device)
    
    real_img = torch.clamp(real_img, 0.0, 1.0)
    fake_img = torch.clamp(fake_img, 0.0, 1.0)

    # SSIM 計算
    ssim_value = ssim(real_img, fake_img, data_range=1.0).item()

    # PSNR 計算
    psnr_value = psnr(real_img, fake_img, data_range=1.0).item()

    # LPIPS 計算
    lpips_value = lpips_fn(real_img, fake_img).mean().item()

    return ssim_value, psnr_value, lpips_value

def calculate_pl(real_img, fake_img, vgg, criterion_perceptual):
    real_img = real_img.to(device)
    fake_img = fake_img.to(device)
    real_features = vgg(real_img)
    fake_features = vgg(fake_img)
    pl_value = sum([criterion_perceptual(f_real, f_fake) for f_real, f_fake in zip(real_features, fake_features)]).item()
    return pl_value

def edge_iou(real_img, fake_img):
    
    if real_img.dim() == 3:  # 單張圖，自動擴維
        real_img = real_img.unsqueeze(0)
        fake_img = fake_img.unsqueeze(0)

    batch_size = real_img.size(0)
    iou_list = []

    for i in range(batch_size):
        real_np = real_img[i].permute(1, 2, 0).detach().cpu().numpy()
        fake_np = fake_img[i].permute(1, 2, 0).detach().cpu().numpy()

        real_gray = cv2.cvtColor((real_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        fake_gray = cv2.cvtColor((fake_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        real_edges = cv2.Canny(real_gray, 100, 200)
        fake_edges = cv2.Canny(fake_gray, 100, 200)

        intersection = np.logical_and(real_edges, fake_edges).sum()
        union = np.logical_or(real_edges, fake_edges).sum()
        iou = intersection / union if union != 0 else 0

        iou_list.append(iou)

    return iou_list if batch_size > 1 else iou_list[0]


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

# === SE Block ===
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

class ResnetBlock(nn.Module):
    """ResNet block with configurable normalization, dropout, and padding."""

    def __init__(self, dim, padding_type='reflect', norm_layer=nn.InstanceNorm2d, use_dropout=False, use_bias=True):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
        self.se_block = SELayer(dim)

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
        out = self.se_block(out)
        return x + out



# === Generator with RES  ===
class Generator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9):
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

# 計算損失函數

def calculate_losses(generator, discriminator, real_img, cycle_img, target_img, vgg, 
                     criterion_gan, criterion_cycle, criterion_perceptual, criterion_identity,
                     frequency_loss, tv_loss, lambda_gan=1.0, lambda_cycle=10.0, lambda_perceptual=0.1, 
                     lambda_identity=0.5, lambda_freq=0.5, use_tv_loss=True, epoch=1):

    fake_img = generator(real_img)
    cycle_generated_img = generator(cycle_img)

    # 對齊圖像尺寸
    real_img, fake_img, cycle_generated_img = align_images(real_img, fake_img, cycle_generated_img)

    # GAN 損失
    gan_loss = criterion_gan(discriminator(fake_img), torch.ones_like(discriminator(fake_img), device=device))

    # 循環一致性損失
    cycle_loss = criterion_cycle(cycle_generated_img, real_img)

    # 感知損失
    real_features = vgg(real_img)
    fake_features = vgg(fake_img)
    cycle_features = vgg(cycle_generated_img)
    perceptual_loss = sum([
        criterion_perceptual(f_real, f_fake) + criterion_perceptual(f_real, f_cycle)
        for f_real, f_fake, f_cycle in zip(real_features, fake_features, cycle_features)
    ])

    # 身份損失
    identity_img = generator(target_img)
    identity_loss = criterion_identity(identity_img, target_img) * lambda_identity

    # 頻譜損失
    freq_loss_weight = min(0.1 + epoch * 0.01, lambda_freq)
    freq_loss = frequency_loss(real_img, fake_img)

    # TV 損失（可選）
    tv_loss_value = tv_loss(fake_img) if use_tv_loss else 0
    
    # 總損失
    total_loss = (lambda_gan * gan_loss + 
                  lambda_cycle * cycle_loss + 
                  lambda_perceptual * perceptual_loss + 
                  identity_loss + 
                  freq_loss_weight * freq_loss + 
                  tv_loss_value)

    return total_loss

def split_train_val_once(rain_root, val_root, ratio=0.3):
    """
    隨機抽樣每個城市 30% 圖片移到 val_root
    """
    if os.path.exists(val_root) and any(os.scandir(val_root)):
        print(f"✔ VAL 資料夾已存在且非空，跳過抽樣分割")
        return

    print(f"🔍 開始分割 train -> val (抽樣比例={ratio*100:.0f}%) ...")

    train_dir = os.path.join(rain_root, 'train')
    os.makedirs(val_root, exist_ok=True)

    for city in os.listdir(train_dir):
        city_train_dir = os.path.join(train_dir, city)
        if not os.path.isdir(city_train_dir):
            continue

        images = [f for f in os.listdir(city_train_dir) if f.endswith('.png')]
        random.shuffle(images)

        num_val = max(1, int(len(images) * ratio))  # 至少抽一張
        val_images = images[:num_val]

        city_val_dir = os.path.join(val_root, city)
        os.makedirs(city_val_dir, exist_ok=True)

        for img in val_images:
            src = os.path.join(city_train_dir, img)
            dst = os.path.join(city_val_dir, img)
            shutil.move(src, dst)

        print(f"城市 {city}: 總數={len(images)}, 移到 val={num_val}")

    print("✅ 抽樣完成！")


class RainToGTDataset(Dataset):
    def __init__(self, rain_root, gt_root, split='train', transform=None):
        """
        rain_root: 雨天圖片資料夾根目錄，例如 "C:/Users/User/Desktop/雨天/leftImg8bit_rain"
        gt_root: GT圖片資料夾根目錄，例如 "C:/Users/User/Desktop/雨天/leftImg8bit_rain/GT"
        split: 'train' 或 'val'，指定要讀哪個子資料夾
        transform: 影像前處理操作（transforms）
        """
        self.rain_dir = os.path.join(rain_root, split)
        self.gt_dir = os.path.join(gt_root)
        self.transform = transform

        self.pairs = []  # 存放 (rain_path, gt_path) 對應

        # 遍歷 rain_dir 所有子資料夾 (城市名)
        for city in os.listdir(self.rain_dir):
            city_rain_dir = os.path.join(self.rain_dir, city)
            city_gt_dir = os.path.join(self.gt_dir, city)
            if not os.path.isdir(city_rain_dir) or not os.path.isdir(city_gt_dir):
                continue

            for fname in os.listdir(city_rain_dir):
                if not fname.endswith(".png"):
                    continue

                # 從雨天檔名萃取前綴字串，範例:
                # aachen_000004_000019_leftImg8bit_rain_alpha_0.01_beta_0.005_dropsize_0.01_pattern_1.png
                # 我們要取 "aachen_000004_000019"
                prefix_match = re.match(r'(.*?)_leftImg8bit', fname)
                if not prefix_match:
                    print(f"[警告] 無法從檔名解析前綴: {fname}")
                    continue

                prefix = prefix_match.group(1) # scene name
                # 對應的 GT 檔名: prefix + "_leftImg8bit.png"
                gt_fname = f"{prefix}_leftImg8bit.png"
                gt_path = os.path.join(city_gt_dir, gt_fname)
                rain_path = os.path.join(city_rain_dir, fname)

                if not os.path.exists(gt_path):
                    print(f"[警告] 找不到對應 GT 檔案: {gt_path}")
                    continue

                self.pairs.append((rain_path, gt_path))

        print(f"\n=== 建立 RainToGTDataset, split={split} ===")

        for rain_path, gt_path in self.pairs:
            rain_name = os.path.basename(rain_path)
            gt_name = os.path.basename(gt_path)
            print(f"[{split}] 配對: {rain_name} ➜ {gt_name}")
        print(f"✔ split={split} 共讀取 {len(self.pairs)} 對\n")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        rain_path, gt_path = self.pairs[idx]

        rain_img = Image.open(rain_path).convert("RGB")
        gt_img = Image.open(gt_path).convert("RGB")

        if self.transform:
            rain_img = self.transform(rain_img)
            gt_img = self.transform(gt_img)

        return rain_img, gt_img, os.path.basename(rain_path)

def train_rain_removal(generator_A2B, generator_B2A, discriminator_A, discriminator_B, train_loader, val_loader, device):
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    train_log_path = os.path.join(log_dir, f"train_results_{timestamp}.txt")
    val_log_path = os.path.join(log_dir, f"val_results_{timestamp}.txt")
    os.makedirs("logs", exist_ok=True)

    criterion_gan = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_perceptual = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    lambda_id = 0.5

    vgg = VGGFeatureExtractor().to(device)
    tv_loss = TVLoss(tv_loss_weight=0.01).to(device)

    optimizer_G = optim.Adam(
        itertools.chain(generator_A2B.parameters(), generator_B2A.parameters()),
        lr=0.0002, betas=(0.5, 0.999)
    )
    optimizer_D_A = optim.Adam(discriminator_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(discriminator_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

    scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=30, gamma=0.5)
    scheduler_D_A = torch.optim.lr_scheduler.StepLR(optimizer_D_A, step_size=30, gamma=0.5)
    scheduler_D_B = torch.optim.lr_scheduler.StepLR(optimizer_D_B, step_size=30, gamma=0.5)

    train_losses, val_losses, epochs_record = [], [], []

    for epoch in range(100):
        epoch_start_time = time.time()
        generator_A2B.train()
        generator_B2A.train()

        epoch_train_loss = 0
        for i, (real_A, real_B, filename) in enumerate(train_loader):
            real_A = real_A.to(device)
            real_B = real_B.to(device)

            fake_B = generator_A2B(real_A)

            # === 計算指標 ===
            ssim_value, psnr_value, lpips_value = calculate_metrics(real_A, fake_B)
            pl_value = calculate_pl(real_A, fake_B, vgg, criterion_perceptual)
            edge_iou_values = edge_iou(real_A, fake_B)
            edge_iou_avg = sum(edge_iou_values) / len(edge_iou_values)
            miou_value = calculate_miou(fake_B, real_A)

            # === log ===
            log_str = (f"Epoch [{epoch+1}/100], Iter [{i+1}] | "
                       f"SSIM: {ssim_value:.4f}, PSNR: {psnr_value:.2f} dB, LPIPS: {lpips_value:.4f}, "
                       f"PL: {pl_value:.4f}, EDGE IoU: {edge_iou_avg:.4f}, mIoU: {miou_value:.4f}")
            print(log_str)
            with open(train_log_path, "a", encoding="utf-8") as f:
                f.write(log_str + "\n")

            # === 損失計算 ===
            optimizer_G.zero_grad()
            loss_G_A2B = calculate_losses(generator_A2B, discriminator_B, real_A, real_B, real_B,
                                          vgg, criterion_gan, criterion_cycle, criterion_perceptual,
                                          criterion_identity, frequency_loss, tv_loss, lambda_id)
            loss_G_B2A = calculate_losses(generator_B2A, discriminator_A, real_B, real_A, real_A,
                                          vgg, criterion_gan, criterion_cycle, criterion_perceptual,
                                          criterion_identity, frequency_loss, tv_loss, lambda_id)
            loss_G = loss_G_A2B + loss_G_B2A
            loss_G.backward()
            optimizer_G.step()

            optimizer_D_A.zero_grad()
            loss_D_A_real = criterion_gan(discriminator_A(real_A), torch.ones_like(discriminator_A(real_A)))
            loss_D_A_fake = criterion_gan(discriminator_A(generator_B2A(real_B).detach()), torch.zeros_like(discriminator_A(real_A)))
            loss_D_A = 0.5 * (loss_D_A_real + loss_D_A_fake)
            loss_D_A.backward()
            optimizer_D_A.step()

            optimizer_D_B.zero_grad()
            loss_D_B_real = criterion_gan(discriminator_B(real_B), torch.ones_like(discriminator_B(real_B)))
            loss_D_B_fake = criterion_gan(discriminator_B(generator_A2B(real_A).detach()), torch.zeros_like(discriminator_B(real_B)))
            loss_D_B = 0.5 * (loss_D_B_real + loss_D_B_fake)
            loss_D_B.backward()
            optimizer_D_B.step()

            epoch_train_loss += loss_G.item()

        scheduler_G.step()
        scheduler_D_A.step()
        scheduler_D_B.step()

        # === 平均 train loss ===
        epoch_train_loss /= max(len(train_loader), 1)
        train_losses.append(epoch_train_loss)
        epochs_record.append(epoch+1)

        val_start_time = time.time() 

        # === Validation ===
        generator_A2B.eval()
        generator_B2A.eval()

        val_loss_epoch = 0
        val_batches = 0
        total_val_ssim = total_val_psnr = total_val_lpips = total_val_miou = 0
        total_val_pl = 0
        total_val_edge_iou = 0

        with torch.no_grad():
            for real_A, real_B, _ in val_loader:
                real_A = real_A.to(device)
                real_B = real_B.to(device)

                loss_G_A2B = calculate_losses(generator_A2B, discriminator_B, real_A, real_B, real_B,
                                            vgg, criterion_gan, criterion_cycle, criterion_perceptual,
                                            criterion_identity, frequency_loss, tv_loss, lambda_id)
                loss_G_B2A = calculate_losses(generator_B2A, discriminator_A, real_B, real_A, real_A,
                                            vgg, criterion_gan, criterion_cycle, criterion_perceptual,
                                            criterion_identity, frequency_loss, tv_loss, lambda_id)
                val_loss_epoch += (loss_G_A2B.item() + loss_G_B2A.item())

                fake_B = generator_A2B(real_A)

                ssim_value, psnr_value, lpips_value = calculate_metrics(real_A, fake_B)
                miou_value = calculate_miou(fake_B, real_A)
                pl_value = calculate_pl(real_A, fake_B, vgg, criterion_perceptual)
                edge_iou_values = edge_iou(real_A, fake_B)
                edge_iou_avg = sum(edge_iou_values) / len(edge_iou_values)

                total_val_ssim += ssim_value
                total_val_psnr += psnr_value
                total_val_lpips += lpips_value
                total_val_miou += miou_value
                total_val_pl += pl_value
                total_val_edge_iou += edge_iou_avg

                val_batches += 1

        avg_val_ssim = total_val_ssim / max(val_batches, 1)
        avg_val_psnr = total_val_psnr / max(val_batches, 1)
        avg_val_lpips = total_val_lpips / max(val_batches, 1)
        avg_val_miou = total_val_miou / max(val_batches, 1)
        avg_val_pl = total_val_pl / max(val_batches, 1)
        avg_val_edge_iou = total_val_edge_iou / max(val_batches, 1)
        val_loss_epoch /= max(val_batches, 1)
        val_losses.append(val_loss_epoch)

        val_elapsed = time.time() - val_start_time
        print(f"Epoch [{epoch+1}/100] 驗證時間: {val_elapsed:.2f} 秒")

        # === 印 & 記錄 val log ===
        val_log_str = (f"Epoch [{epoch+1}/100], Val Loss: {val_loss_epoch:.4f}, "
                    f"Val SSIM: {avg_val_ssim:.4f}, Val PSNR: {avg_val_psnr:.2f}, "
                    f"Val LPIPS: {avg_val_lpips:.4f}, Val PL: {avg_val_pl:.4f}, "
                    f"Edge IoU: {avg_val_edge_iou:.4f}, Val mIoU: {avg_val_miou:.4f}, "
                    f"Val Time: {val_elapsed:.2f}s")
        print(val_log_str)

        with open(val_log_path, "a", encoding="utf-8") as f:
            f.write(val_log_str + "\n")

        # === 輸出 CSV ===
        csv_path = os.path.join(base_dir, "loss_log.csv")
        with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss"])
            for epoch_num, train_loss, val_loss in zip(epochs_record, train_losses, val_losses):
                writer.writerow([epoch_num, train_loss, val_loss])

        print(f"✔ 損失資料已儲存至 {csv_path}")


        # === 模型儲存 ===
        if (epoch + 1) % 1 == 0:
            elapsed_time = time.time() - epoch_start_time
            print(f"Epoch [{epoch+1}/100] 訓練時間: {elapsed_time:.2f} 秒")
            if torch.cuda.is_available():
                allocated_memory = torch.cuda.memory_allocated(device) / 1024 ** 2
                reserved_memory = torch.cuda.memory_reserved(device) / 1024 ** 2
                print(f"GPU 記憶體使用量: 已分配 {allocated_memory:.2f} MB, 已預留 {reserved_memory:.2f} MB")

            os.makedirs(os.path.join(base_dir, 'CycleGAN_Results'), exist_ok=True)
            save_path = os.path.join(base_dir, f"epoch{epoch + 1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'generator_A2B': generator_A2B.state_dict(),
                'generator_B2A': generator_B2A.state_dict(),
                'discriminator_A': discriminator_A.state_dict(),
                'discriminator_B': discriminator_B.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D_A': optimizer_D_A.state_dict(),
                'optimizer_D_B': optimizer_D_B.state_dict(),
            }, save_path)
            print(f"✔ 模型已儲存至 {save_path}")

        print(f"Epoch [{epoch+1}/100], Loss G_A2B: {loss_G_A2B.item()} Loss G_B2A: {loss_G_B2A.item()} Loss D_A: {loss_D_A.item()} Loss D_B: {loss_D_B.item()} "
              f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {val_loss_epoch:.4f}")
        
        epoch_elapsed = time.time() - epoch_start_time
        print(f"Epoch [{epoch+1}/100] 單次訓練時間: {epoch_elapsed:.2f} 秒")

        with open(train_log_path, "a", encoding="utf-8") as f:
            f.write(f"Epoch [{epoch+1}/100] 單次訓練時間: {epoch_elapsed:.2f} 秒\n")
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

train_dataset = RainToGTDataset(
    rain_root=r'C:\Users\User\Desktop\小城市測試\leftImg8bit_rain',
    gt_root=r'C:\Users\User\Desktop\小城市測試\leftImg8bit_rain\GT',
    split='train',  # 或 'val'
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

def test_folder_images(generator, input_folder, output_folder, device):
    generator.eval()
    transform_test = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    os.makedirs(output_folder, exist_ok=True)

    with torch.no_grad():
        for fname in os.listdir(input_folder):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            input_path = os.path.join(input_folder, fname)
            output_path = os.path.join(output_folder, fname)

            img = Image.open(input_path).convert("RGB")
            img_tensor = transform_test(img).unsqueeze(0).to(device)

            output = generator(img_tensor)
            output_upscaled = F.interpolate(output, size=(256, 512), mode='bilinear', align_corners=False)

            save_image(output_upscaled, output_path)
            print(f"✔ 已儲存：{output_path}")


if __name__ == "__main__":
    rain_root = r'C:\Users\User\Desktop\小城市測試\leftImg8bit_rain'
    gt_root = os.path.join(rain_root, 'GT')
    val_root = os.path.join(rain_root, 'VAL')

    # Step 1: 先做一次抽樣移動
    split_train_val_once(rain_root, val_root, ratio=0.3)

    # Step 2: 建立 DataLoader
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    train_dataset = RainToGTDataset(rain_root, gt_root, split='train', transform=transform)
    val_dataset = RainToGTDataset(rain_root, gt_root, split='VAL', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    print(f"✔ train_loader 長度: {len(train_loader)}")
    print(f"✔ val_loader 長度: {len(val_loader)}")

    # 原本的模型與訓練
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator_A2B = Generator().to(device)
    generator_B2A = Generator().to(device)
    PatchGANDiscriminator_A = SpectralPatchGANDiscriminator().to(device)
    PatchGANDiscriminator_B = SpectralPatchGANDiscriminator().to(device)

    # 開始訓練
    train_rain_removal(generator_A2B, generator_B2A, PatchGANDiscriminator_A, PatchGANDiscriminator_B, train_loader, val_loader, device)
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device)

# 你想要儲存的資料夾路徑
base_dir = r'C:\Users\User\Desktop\魏梓祐注意力機制\SE+CBAM\model\\'

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

def compute_flops_params(model, input_shape=(1, 3, 256, 512)):
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

class PAModule(nn.Module):
    def __init__(self, in_channels):
        super(PAModule, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.size()
        query = self.query_conv(x).view(b, -1, w * h).permute(0, 2, 1)  # (b, N, C')
        key = self.key_conv(x).view(b, -1, w * h)  # (b, C', N)
        energy = torch.bmm(query, key)  # (b, N, N)
        attention = self.softmax(energy)  # 計算注意力權重
        value = self.value_conv(x).view(b, -1, w * h)  # (b, C, N)

        out = torch.bmm(value, attention.permute(0, 2, 1))  # (b, C, N)
        out = out.view(b, c, h, w)  # 變回 (b, C, H, W)
        return x + out  # 加回原特徵

class CAModule(nn.Module):
    def __init__(self, in_channels):
        super(CAModule, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        b, c, h, w = x.size()
        query = self.query_conv(x).view(b, c, -1)  # (b, C, N)
        key = self.key_conv(x).view(b, c, -1).permute(0, 2, 1)  # (b, N, C)
        energy = torch.bmm(query, key)  # (b, C, C)
        attention = self.softmax(energy)  # 計算注意力權重
        value = self.value_conv(x).view(b, c, -1)  # (b, C, N)

        out = torch.bmm(attention, value)  # (b, C, N)
        out = out.view(b, c, h, w)  # 變回 (b, C, H, W)
        return x + out  # 加回原特徵

class DANet(nn.Module):
    def __init__(self, in_channels):
        super(DANet, self).__init__()
        self.pa = PAModule(in_channels)
        self.ca = CAModule(in_channels)

    def forward(self, x):
        pa_out = self.pa(x)  # 位置注意力
        ca_out = self.ca(x)  # 通道注意力
        return pa_out + ca_out  # 加總兩個注意力

# === CSP 殘差塊 + DANET ===
class CSPResBlockWithDANet(nn.Module):
    def __init__(self, channels):
        super(CSPResBlockWithDANet, self).__init__()
        self.split_channels = channels // 2

        self.main_path = nn.Sequential(
            nn.Conv2d(self.split_channels, self.split_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(self.split_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.split_channels, self.split_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(self.split_channels),
        )

        self.da = DANet(self.split_channels)  # 使用 Dual Attention 模組

        self.final_conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        out = self.main_path(x1)
        out = self.da(out)  # 應用 DANet 注意力
        out = x1 + 0.7 * out
        out = torch.cat([out, x2], dim=1)
        return self.final_conv(out)

# === Generator with CSP + DANET ===
class GeneratorWithCSPAttention(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, num_res_blocks=6, width=512, height=256):
        super(GeneratorWithCSPAttention, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.width = width
        self.height = height
        self.res_blocks = nn.Sequential(*[CSPResBlockWithDANet(256) for _ in range(num_res_blocks)])

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, output_channels, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        x = F.interpolate(x, size=(self.height, self.width))
        x = self.encoder(x)
        x = self.res_blocks(x)
        x = self.decoder(x)
        return x


# **多尺度 PatchGAN 判別器**
class MultiScalePatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3, num_scales=2):
        super(MultiScalePatchGANDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            SpectralPatchGANDiscriminator(in_channels) for _ in range(num_scales)
        ])
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        outputs = []
        for d in self.discriminators:
            outputs.append(d(x))
            x = self.downsample(x)
        return outputs

# **改進版 PatchGAN 判別器**
class SpectralPatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(SpectralPatchGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            SpectralNormConv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNormConv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNormConv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNormConv2d(256, 512, kernel_size=4, stride=1, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)

# TVLoss
class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=0.01):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).mean()
        w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2).mean()
        return self.tv_loss_weight * (h_tv + w_tv)
   
# **VGG 特徵提取器 (支持多層特徵)**
class VGGFeatureExtractor(nn.Module):
    def __init__(self, layers=[2, 7, 12, 21]):
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

def frequency_loss(real_img, fake_img, eps=1e-8):
    real_fft = torch.fft.fft2(real_img, dim=(-2, -1))
    fake_fft = torch.fft.fft2(fake_img, dim=(-2, -1))
    loss = nn.L1Loss()(torch.abs(real_fft) + eps, torch.abs(fake_fft) + eps)
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

def train_rain_removal(generator_A2B, generator_B2A, discriminator_A, discriminator_B, paired_loader, device):
    criterion_gan = nn.MSELoss()  # 對抗損失
    criterion_cycle = nn.L1Loss()  # 循環一致性損失
    criterion_perceptual = nn.L1Loss()  # 感知損失
    criterion_identity = nn.L1Loss()  # 身份損失
    lambda_id = 0.5

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

    start_time = time.time()  # 記錄開始時間
    log_path = os.path.join(base_dir, "results.txt")

    for epoch in range(150):
        for i, (real_A, real_B, filename) in enumerate(paired_loader):
            real_A = real_A.to(device)
            real_B = real_B.to(device)
            real_A = F.interpolate(real_A, size=(256, 512))
            real_B = F.interpolate(real_B, size=(256, 512))
            fake_B = generator_A2B(real_A)
            
            # 計算 FLOPs & Params（只需計算一次，存儲結果）
            if epoch == 0 and i == 0:
                flops, params = compute_flops_params(generator_A2B)

            # 計算 SSIM、PSNR 和 LPIPS（修改：每次迭代都計算）
            ssim_value, psnr_value, lpips_value = calculate_metrics(real_A, fake_B)
            pl_value = calculate_pl(real_A, fake_B, vgg, criterion_perceptual)

            #改：edge_iou 現在支援 batch，回傳 list
            edge_iou_values = edge_iou(real_A, fake_B)
            edge_iou_avg = sum(edge_iou_values) / len(edge_iou_values)

            miou_value = calculate_miou(fake_B, real_A)  # 確保 real_A 是 segmentation mask

            # 格式化輸出字串
            log_str = (f"Epoch [{epoch+1}/150], Iter [{i+1}] | "
                    f"SSIM: {ssim_value:.4f}, PSNR: {psnr_value:.2f} dB, LPIPS: {lpips_value:.4f}, "
                    f"PL: {pl_value:.4f}, EDGE IoU: {edge_iou_avg:.4f}, mIoU: {miou_value:.4f}, "
                    f"FLOPs: {flops/1e9:.2f}G, Params: {params/1e6:.2f}M")

            # 印出到終端
            print(log_str)

            # 每次都寫入總檔案
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(log_str + "\n")

            # 每 30 epoch 另存一份獨立 TXT
            if (epoch + 1) % 30 == 0:
                epoch_txt_path = os.path.join(base_dir, f"epoch{epoch + 1}.txt")
                with open(epoch_txt_path, "a", encoding="utf-8") as f:
                    f.write(log_str + "\n")


            
            min_height = min(real_A.size(2), real_B.size(2), fake_B.size(2))
            min_width = min(real_A.size(3), real_B.size(3), fake_B.size(3))

            real_A = real_A[:, :, :min_height, :min_width]
            real_B = real_B[:, :, :min_height, :min_width]
            fake_B = fake_B[:, :, :min_height, :min_width]

            optimizer_G.zero_grad()
            
            loss_G_A2B = calculate_losses(generator_A2B, discriminator_B, real_A, real_B, real_B,
                                          vgg, criterion_gan, criterion_cycle, criterion_perceptual, criterion_identity,
                                          frequency_loss, tv_loss, lambda_id)
            loss_G_B2A = calculate_losses(generator_B2A, discriminator_A, real_B, real_A, real_A,
                                          vgg, criterion_gan, criterion_cycle, criterion_perceptual, criterion_identity,
                                          frequency_loss, tv_loss, lambda_id)

            loss_G = loss_G_A2B + loss_G_B2A
            loss_G.backward()
            optimizer_G.step()

            optimizer_D_A.zero_grad()
            loss_D_A_real = criterion_gan(discriminator_A(real_A), torch.ones_like(discriminator_A(real_A), device=device))
            loss_D_A_fake = criterion_gan(discriminator_A(generator_B2A(real_B).detach()), torch.zeros_like(discriminator_A(real_A), device=device))
            loss_D_A = 0.5 * (loss_D_A_real + loss_D_A_fake)
            loss_D_A.backward()
            optimizer_D_A.step()

            optimizer_D_B.zero_grad()
            loss_D_B_real = criterion_gan(discriminator_B(real_B), torch.ones_like(discriminator_B(real_B), device=device))
            loss_D_B_fake = criterion_gan(discriminator_B(generator_A2B(real_A).detach()), torch.zeros_like(discriminator_B(real_B), device=device))
            loss_D_B = 0.5 * (loss_D_B_real + loss_D_B_fake)
            loss_D_B.backward()
            optimizer_D_B.step()

        scheduler_G.step()
        scheduler_D_A.step()
        scheduler_D_B.step()

        # **計算與列印訓練時間與記憶體使用量 (每 30 epoch)**
        if (epoch + 1) % 30 == 0:
            elapsed_time = time.time() - start_time  # 計算總時間
            print(f"Epoch [{epoch+1}/150] 訓練時間: {elapsed_time:.2f} 秒")

            if torch.cuda.is_available():
                allocated_memory = torch.cuda.memory_allocated(device) / 1024 ** 2  # MB
                reserved_memory = torch.cuda.memory_reserved(device) / 1024 ** 2  # MB
                print(f"GPU 記憶體使用量: 已分配 {allocated_memory:.2f} MB, 已預留 {reserved_memory:.2f} MB")

        os.makedirs(os.path.join(base_dir, 'CycleGAN_Results'), exist_ok=True)

        if (epoch + 1) % 30 == 0:
            os.makedirs(base_dir, exist_ok=True)  # 確保資料夾存在
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
            
        print(f"Epoch [{epoch+1}/{150}], Iter [{i+1}] Loss G_A2B: {loss_G_A2B.item()} Loss G_B2A: {loss_G_B2A.item()} Loss D_A: {loss_D_A.item()} Loss D_B: {loss_D_B.item()}")

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

train_dataset = RainToGTDataset(
    rain_root=r"C:/Users/User/Desktop/雨天/leftImg8bit_rain",
    gt_root=r"C:/Users/User/Desktop/雨天/leftImg8bit_rain/GT",
    split='train',  # 或 'val'
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

if __name__ == "__main__":
    # 設置模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator_A2B = GeneratorWithCSPAttention(width=512, height=256).to(device)
    generator_B2A = GeneratorWithCSPAttention(width=512, height=256).to(device)
    PatchGANDiscriminator_A = SpectralPatchGANDiscriminator().to(device)
    PatchGANDiscriminator_B = SpectralPatchGANDiscriminator().to(device)

    # 開始訓練
    train_rain_removal(generator_A2B, generator_B2A, PatchGANDiscriminator_A, PatchGANDiscriminator_B, train_loader, device)


    output_dir = r'C:\Users\User\Desktop\魏梓祐注意力機制\SE+CBAM\R'
    os.makedirs(output_dir, exist_ok=True)

    for i, (real_A, real_B, filename) in enumerate(train_loader):
        real_A = real_A.to(device)
        fake_B = generator_A2B(real_A)

        # 保存生成影像，保留原始檔名
        save_path = os.path.join(output_dir, filename[0])
        save_image(fake_B.detach().cpu() * 0.5 + 0.5, save_path)
        print(f"已保存生成影像：{save_path}")

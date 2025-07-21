# main_unpaired.py
# ✅ 改為 unpaired 訓練版本：刪除 val、使用隨機 unpaired trainA/trainB 資料訓練
# ✅ 保留你原本的 loss 設計（GAN + cycle + identity + perceptual + freq + TV）
# ✅ 保留配對的 RainToGTDataset class，供 test 使用

import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from main import Generator, SpectralPatchGANDiscriminator, calculate_losses, VGGFeatureExtractor, TVLoss, frequency_loss
import torch.nn as nn
import torch.optim as optim
import itertools
import time

save_dir = r"C:\Users\User\Desktop\小城市測試\models"
os.makedirs(save_dir, exist_ok=True)

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

# === 訓練函數（不含 val） ===
def train_cyclegan_unpaired(generator_A2B, generator_B2A, discriminator_A, discriminator_B, dataloader, device):
    criterion_gan = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_perceptual = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    vgg = VGGFeatureExtractor().to(device)
    tv_loss = TVLoss(tv_loss_weight=0.01).to(device)

    optimizer_G = optim.Adam(
        itertools.chain(generator_A2B.parameters(), generator_B2A.parameters()),
        lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_A = optim.Adam(discriminator_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(discriminator_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(150):
        generator_A2B.train()
        generator_B2A.train()

        epoch_loss = 0
        for i, (real_A, real_B) in enumerate(dataloader):
            real_A = real_A.to(device)
            real_B = real_B.to(device)

            optimizer_G.zero_grad()
            loss_G_A2B = calculate_losses(generator_A2B, discriminator_B, real_A, real_B, real_B,
                                          vgg, criterion_gan, criterion_cycle, criterion_perceptual,
                                          criterion_identity, frequency_loss, tv_loss, lambda_identity=0.5, epoch=epoch)
            loss_G_B2A = calculate_losses(generator_B2A, discriminator_A, real_B, real_A, real_A,
                                          vgg, criterion_gan, criterion_cycle, criterion_perceptual,
                                          criterion_identity, frequency_loss, tv_loss, lambda_identity=0.5, epoch=epoch)
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

            epoch_loss += loss_G.item()

        print(f"Epoch [{epoch+1}/150], Loss_G: {epoch_loss:.4f}")

        torch.save({
            'generator_A2B': generator_A2B.state_dict(),
            'generator_B2A': generator_B2A.state_dict(),
            'discriminator_A': discriminator_A.state_dict(),
            'discriminator_B': discriminator_B.state_dict()
        }, os.path.join(save_dir, f"checkpoint_epoch{epoch+1}.pth"))
        
        print(f"✔ 模型已儲存於 checkpoint_epoch{epoch+1}.pth")

# === 主程式 ===
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rain_root = r'C:\Users\User\Desktop\小城市測試\leftImg8bit_rain\trainA'
    sun_root = r'C:\Users\User\Desktop\小城市測試\leftImg8bit_rain\trainB'

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    train_dataset = UnpairedImageDataset(rain_root, sun_root, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

    generator_A2B = Generator().to(device)
    generator_B2A = Generator().to(device)
    discriminator_A = SpectralPatchGANDiscriminator().to(device)
    discriminator_B = SpectralPatchGANDiscriminator().to(device)

    train_cyclegan_unpaired(generator_A2B, generator_B2A, discriminator_A, discriminator_B, train_loader, device)

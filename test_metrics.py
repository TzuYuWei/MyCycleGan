# test_metrics.py
# ✅ 測試模型並配對 GT，計算 SSIM、PSNR、LPIPS、PL、EDGE IoU（OpenCV）、mIoU + FLOPs/Params

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from piq import ssim, psnr
import lpips
from torchmetrics import JaccardIndex
from torchvision.models import vgg19, VGG19_Weights, inception_v3
import torch.nn.functional as F
import cv2
import numpy as np
from thop import profile
TXT_dir = r'C:\Users\User\Desktop\CycleGan_128\result\train_mean'

# === 測試資料集 ===
class RainToGTDataset(Dataset):
    def __init__(self, rain_root, gt_root, transform=None):
        self.rain_paths = []
        self.gt_paths = []
        self.transform = transform

        for city in os.listdir(rain_root):
            rain_city_path = os.path.join(rain_root, city)
            gt_city_path = os.path.join(gt_root, city)
            if not os.path.isdir(gt_city_path):
                continue

            rain_files = sorted([f for f in os.listdir(rain_city_path) if f.endswith('.png')])
            for rain_name in rain_files:
                base_name = rain_name.split("_rain")[0] + ".png"
                gt_path = os.path.join(gt_city_path, base_name)
                rain_path = os.path.join(rain_city_path, rain_name)
                if os.path.exists(gt_path):
                    self.rain_paths.append(rain_path)
                    self.gt_paths.append(gt_path)

    def __len__(self):
        return len(self.rain_paths)

    def __getitem__(self, idx):
        rain_img = Image.open(self.rain_paths[idx]).convert("RGB")
        gt_img = Image.open(self.gt_paths[idx]).convert("RGB")
        if self.transform:
            rain_img = self.transform(rain_img)
            gt_img = self.transform(gt_img)
        name = os.path.basename(self.rain_paths[idx])
        gt_name = os.path.basename(self.gt_paths[idx])
        return rain_img, gt_img, name, gt_name

# === Perceptual Loss ===
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:9]
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.eval()

    def forward(self, x, y):
        return F.l1_loss(self.vgg(x), self.vgg(y))

# === OpenCV Edge IoU ===
def edge_iou_opencv(real_img, fake_img):
    if real_img.dim() == 3:
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

    return iou_list[0] if batch_size == 1 else iou_list

# === FLOPs and Params ===
def compute_flops_params(model, input_shape=(1, 3, 128, 128), device='cpu'):
    dummy_input = torch.randn(input_shape).to(device)
    flops, params = profile(model, inputs=(dummy_input,))
    return flops, params

# === 主測試函數 ===
def test_model(generator, dataloader, device, save_dir, TXT_dir):
    generator.eval()
    os.makedirs(TXT_dir, exist_ok=True)
    result_txt = os.path.join(TXT_dir, "test_results.txt")

    lpips_fn = lpips.LPIPS(net='alex').to(device)
    perceptual_loss_fn = VGGPerceptualLoss().to(device)
    jaccard = JaccardIndex(task="binary").to(device)

    total_ssim = total_psnr = total_lpips = total_pl = total_edge_iou = total_miou = 0

    # 計算 FLOPs 和參數量
    flops, params = compute_flops_params(generator, device=device)
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
    print(f"Params: {params / 1e6:.2f} M")

    with open(result_txt, "w") as log_file:
        with torch.no_grad():
            for i, (rain_img, gt_img, name, gt_name) in enumerate(dataloader):
                rain_img = rain_img.to(device)
                gt_img = gt_img.to(device)

                fake_sunny = generator(rain_img)
                fake_sunny = F.interpolate(fake_sunny, size=(128, 128), mode='bilinear', align_corners=False)
                gt_img = F.interpolate(gt_img, size=(128, 128), mode='bilinear', align_corners=False)
                fake_sunny = fake_sunny.clamp(0.0, 1.0)
                gt_img = gt_img.clamp(0.0, 1.0)
                # 🧩 加這段來確保 name 是字串
                if isinstance(name, (list, tuple)):
                    name = name[0]

                # 儲存圖片
                save_path = os.path.join(save_dir, f"fake_{name}")
                save_image(fake_sunny, save_path)

                ssim_val = ssim(fake_sunny, gt_img, data_range=1.0).item()
                psnr_val = psnr(fake_sunny, gt_img, data_range=1.0).item()
                lpips_val = lpips_fn(fake_sunny, gt_img).mean().item()
                pl_val = perceptual_loss_fn(fake_sunny, gt_img).item()
                edge_val = edge_iou_opencv(gt_img, fake_sunny)

                pred_bin = (fake_sunny.mean(dim=1, keepdim=True) > 0.5).int()
                target_bin = (gt_img.mean(dim=1, keepdim=True) > 0.5).int()
                miou_val = jaccard(pred_bin, target_bin).item()

                total_ssim += ssim_val
                total_psnr += psnr_val
                total_lpips += lpips_val
                total_pl += pl_val
                total_edge_iou += edge_val
                total_miou += miou_val

                # 儲存指標 log
                print(f"[配對] {name} ➜ {gt_name}")
                log_file.write(f"{name}, SSIM: {ssim_val:.4f}, PSNR: {psnr_val:.2f} dB, LPIPS: {lpips_val:.4f}, PL: {pl_val:.4f}, EDGE IoU: {edge_val:.4f}, mIoU: {miou_val:.4f}\n")
'''
        n = len(dataloader)
        print("===== 評估指標結果（平均） =====")
        print(f"SSIM: {total_ssim / n:.4f}")
        print(f"PSNR: {total_psnr / n:.2f} dB")
        print(f"LPIPS: {total_lpips / n:.4f}")
        print(f"Perceptual Loss: {total_pl / n:.4f}")
        print(f"EDGE IoU (OpenCV): {total_edge_iou / n:.4f}")
        print(f"mIoU: {total_miou / n:.4f}")
'''
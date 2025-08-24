# test_metrics_from_folder.py
# ✅ 使用「現成的生成圖片資料夾」配對 GT，計算 SSIM、PSNR、LPIPS、PL、EDGE IoU

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from piq import ssim, psnr
import lpips
from torchvision.models import vgg19, VGG19_Weights
import torch.nn.functional as F
import cv2
import numpy as np

# === Dataset（讀取生成圖片 + GT）===
class GenToGTDataset(Dataset):
    def __init__(self, gen_root, gt_root, transform=None):
        self.gen_paths = []
        self.gt_paths = []
        self.transform = transform

        # 遞迴找生成圖片
        gen_files = []
        for root, _, files in os.walk(gen_root):
            for f in files:
                if f.endswith('.png'):
                    gen_files.append(os.path.join(root, f))

        for gen_path in sorted(gen_files):
            gen_name = os.path.basename(gen_path)

            # 去掉 rain 後綴（如果有）
            if "_rain_" in gen_name:
                base_name = gen_name.split("_rain_")[0] + ".png"
            else:
                base_name = gen_name

            # 根據檔名前綴找城市
            city_name = base_name.split("_")[0]
            gt_city_path = os.path.join(gt_root, city_name)
            gt_path = os.path.join(gt_city_path, base_name)

            if os.path.exists(gt_path):
                self.gen_paths.append(gen_path)
                self.gt_paths.append(gt_path)
            else:
                print(f"找不到對應 GT：{gt_path}")

    def __len__(self):
        return len(self.gen_paths)

    def __getitem__(self, idx):
        gen_img = Image.open(self.gen_paths[idx]).convert("RGB")
        gt_img = Image.open(self.gt_paths[idx]).convert("RGB")
        if self.transform:
            gen_img = self.transform(gen_img)
            gt_img = self.transform(gt_img)
        name = os.path.basename(self.gen_paths[idx])
        return gen_img, gt_img, name

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

# === Edge IoU ===
def edge_iou_opencv(real_img, fake_img):
    real_np = real_img.permute(1, 2, 0).detach().cpu().numpy()
    fake_np = fake_img.permute(1, 2, 0).detach().cpu().numpy()

    real_gray = cv2.cvtColor((real_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    fake_gray = cv2.cvtColor((fake_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    real_edges = cv2.Canny(real_gray, 100, 200)
    fake_edges = cv2.Canny(fake_gray, 100, 200)

    intersection = np.logical_and(real_edges, fake_edges).sum()
    union = np.logical_or(real_edges, fake_edges).sum()
    return intersection / union if union != 0 else 0


# === 主測試函數（不需 generator）===
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image

def test_generated_images(dataloader, device, TXT_dir, quantize=False):
    os.makedirs(TXT_dir, exist_ok=True)
    result_txt = os.path.join(TXT_dir, f"test_results_SECBAM3.txt")

    lpips_fn = lpips.LPIPS(net='alex').to(device)
    perceptual_loss_fn = VGGPerceptualLoss().to(device)

    total_ssim = total_psnr = total_lpips = total_pl = total_edge_iou = 0

    # 暫存資料夾（用於 quantize 模式）
    tmp_dir = os.path.join(TXT_dir, "tmp_eval")
    if quantize:
        os.makedirs(tmp_dir, exist_ok=True)

    with open(result_txt, "w", encoding="utf-8") as log_file:
        with torch.no_grad():
            for i, (gen_img, gt_img, name) in enumerate(dataloader):
                if isinstance(name, (list, tuple)):
                    name = name[0]
                gen_img = gen_img.to(device)
                gt_img = gt_img.to(device)

                gen_img = F.interpolate(gen_img, size=(256, 256), mode='bilinear', align_corners=False)
                gt_img = F.interpolate(gt_img, size=(256, 256), mode='bilinear', align_corners=False)
                gen_img = gen_img.clamp(0.0, 1.0)
                gt_img = gt_img.clamp(0.0, 1.0)

                # === quantize 模式：存成 PNG 再讀回來 ===
                if quantize:
                    if isinstance(name, (list, tuple)):
                        name = name[0]  # 🔧 解 tuple
                    tmp_path = os.path.join(tmp_dir, name)
                    save_image(gen_img, tmp_path)
                    gen_img = transforms.ToTensor()(Image.open(tmp_path).convert("RGB")).unsqueeze(0).to(device)
                # ======================================


                ssim_val = ssim(gen_img, gt_img, data_range=1.0).item()
                psnr_val = psnr(gen_img, gt_img, data_range=1.0).item()
                lpips_val = lpips_fn(gen_img, gt_img).mean().item()
                pl_val = perceptual_loss_fn(gen_img, gt_img).item()
                edge_val = edge_iou_opencv(gt_img[0], gen_img[0])

                total_ssim += ssim_val
                total_psnr += psnr_val
                total_lpips += lpips_val
                total_pl += pl_val
                total_edge_iou += edge_val

                # 儲存指標 log
                print(f"[配對] {name}")
                log_file.write(f"{name}, SSIM: {ssim_val:.4f}, PSNR: {psnr_val:.2f} dB, LPIPS: {lpips_val:.4f}, PL: {pl_val:.4f}, EDGE IoU: {edge_val:.4f}\n")



# test_with_metrics_auto_gt.py

from main import Generator, VGGFeatureExtractor, calculate_metrics, calculate_pl, edge_iou, calculate_miou
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os
import re

# === 設定路徑 ===
model_path = r'model路徑'            # 你的模型.pth
test_rain_root = r'test_rain資料夾'   # ex: C:\...\leftImg8bit_rain\test
test_gt_root   = r'GT資料夾'          # ex: C:\...\leftImg8bit_rain\GT
output_dir = r'C:\Users\User\Desktop\小城市測試\test_output'
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 載入模型 ===
generator = Generator().to(device)
checkpoint = torch.load(model_path, map_location=device)
generator.load_state_dict(checkpoint['generator_A2B'])
generator.eval()
print(f"✔ 已載入模型：{model_path}")

# === VGG & 感知損失 ===
vgg = VGGFeatureExtractor().to(device)
criterion_perceptual = nn.L1Loss()

# === transform ===
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# === 累計變數 ===
total_ssim = total_psnr = total_lpips = total_pl = total_edge_iou = total_miou = 0
count = 0

# 遍歷所有城市子資料夾
for city in os.listdir(test_rain_root):
    rain_city_dir = os.path.join(test_rain_root, city)
    gt_city_dir = os.path.join(test_gt_root, city)
    if not os.path.isdir(rain_city_dir):
        continue

    for fname in os.listdir(rain_city_dir):
        if not fname.endswith(".png"):
            continue

        rain_path = os.path.join(rain_city_dir, fname)

        # 前綴擷取
        prefix_match = re.match(r"(.*?)_leftImg8bit", fname)
        if not prefix_match:
            print(f"[警告] 無法擷取前綴: {fname}")
            continue

        prefix = prefix_match.group(1)
        gt_fname = f"{prefix}_leftImg8bit.png"
        gt_path = os.path.join(gt_city_dir, gt_fname)

        if not os.path.exists(gt_path):
            print(f"[警告] 找不到 GT 圖片: {gt_path}")
            continue

        # 載入與轉換影像
        rain_img = transform(Image.open(rain_path).convert("RGB")).unsqueeze(0).to(device)
        gt_img = transform(Image.open(gt_path).convert("RGB")).unsqueeze(0).to(device)

        # 生成影像
        with torch.no_grad():
            fake_img = generator(rain_img)

        # 儲存影像
        save_path = os.path.join(output_dir, f"{city}_{fname}")
        save_image(fake_img, save_path)

        # 計算指標
        ssim_v, psnr_v, lpips_v = calculate_metrics(gt_img, fake_img)
        pl_v = calculate_pl(gt_img, fake_img, vgg, criterion_perceptual)
        edge_iou_v = edge_iou(gt_img, fake_img)
        miou_v = calculate_miou(fake_img, gt_img)

        # 累加
        total_ssim += ssim_v
        total_psnr += psnr_v
        total_lpips += lpips_v
        total_pl += pl_v
        total_edge_iou += edge_iou_v if isinstance(edge_iou_v, float) else sum(edge_iou_v) / len(edge_iou_v)
        total_miou += miou_v
        count += 1

        print(f"✔ [{count}] {city}/{fname} 測試完成")

# === 平均輸出 ===
if count > 0:
    print("\n=== 平均測試結果 ===")
    print(f"SSIM:       {total_ssim / count:.4f}")
    print(f"PSNR:       {total_psnr / count:.2f} dB")
    print(f"LPIPS:      {total_lpips / count:.4f}")
    print(f"Perceptual: {total_pl / count:.4f}")
    print(f"EDGE IoU:   {total_edge_iou / count:.4f}")
    print(f"mIoU:       {total_miou / count:.4f}")
else:
    print("⚠️ 沒有成功測試任何圖片。")

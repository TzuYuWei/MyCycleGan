import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vgg19, VGG19_Weights
from PIL import Image
import lpips
import cv2
import numpy as np
import pandas as pd
from piq import ssim, psnr  # pip install piq


# === Dataset（生成圖片對單一 GT）===
class GenToOneGTDataset(Dataset):
    def __init__(self, gen_root, gt_path, transform=None):
        self.gen_paths = []
        self.gt_path = gt_path
        self.transform = transform

        # 收集生成圖片
        for root, _, files in os.walk(gen_root):
            for f in files:
                if f.endswith('.bmp'):
                    self.gen_paths.append(os.path.join(root, f))

        self.gen_paths = sorted(self.gen_paths)

    def __len__(self):
        return len(self.gen_paths)

    def __getitem__(self, idx):
        gen_img = Image.open(self.gen_paths[idx]).convert("RGB")
        gt_img = Image.open(self.gt_path).convert("RGB")
        if self.transform:
            gen_img = self.transform(gen_img)
            gt_img = self.transform(gt_img)
        name = os.path.basename(self.gen_paths[idx])
        return gen_img, gt_img, name


# === VGG 感知損失 ===
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


# === 主測試函數（輸出CSV）===
def test_generated_images(dataloader, device, CSV_dir):
    os.makedirs(CSV_dir, exist_ok=True)
    result_csv = os.path.join(CSV_dir, "test_results.csv")

    lpips_fn = lpips.LPIPS(net='alex').to(device)
    perceptual_loss_fn = VGGPerceptualLoss().to(device)

    records = []

    with torch.no_grad():
        for i, (gen_img, gt_img, name) in enumerate(dataloader):
            gen_img = gen_img.to(device)
            gt_img = gt_img.to(device)

            # resize 與 clamp
            gen_img = F.interpolate(gen_img, size=(256, 256), mode='bilinear', align_corners=False)
            gt_img = F.interpolate(gt_img, size=(256, 256), mode='bilinear', align_corners=False)
            gen_img = gen_img.clamp(0.0, 1.0)
            gt_img = gt_img.clamp(0.0, 1.0)

            # === 計算指標 ===
            ssim_val = ssim(gen_img, gt_img, data_range=1.0).item()
            psnr_val = psnr(gen_img, gt_img, data_range=1.0).item()
            lpips_val = lpips_fn(gen_img, gt_img).mean().item()
            pl_val = perceptual_loss_fn(gen_img, gt_img).item()
            edge_val = edge_iou_opencv(gt_img[0], gen_img[0])

            records.append({
                "name": name,
                "SSIM": ssim_val,
                "PSNR": psnr_val,
                "LPIPS": lpips_val,
                "PL": pl_val,
                "Edge_IoU": edge_val
            })

            print(f"[配對] {name} 完成")

    # === 平均值 ===
    df = pd.DataFrame(records)
    mean_row = {
        "name": "Average",
        "SSIM": df["SSIM"].mean(),
        "PSNR": df["PSNR"].mean(),
        "LPIPS": df["LPIPS"].mean(),
        "PL": df["PL"].mean(),
        "Edge_IoU": df["Edge_IoU"].mean()
    }
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    # 存成CSV
    df.to_csv(result_csv, index=False, encoding="utf-8-sig")
    print(f"✅ 結果已存到 {result_csv}")


# === 使用範例 ===
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    gen_root = r"C:\Users\ericw\Desktop\bmp2\munster_000121_000019_leftImg8bit_rain_alpha_0.01_beta_0.005_dropsize_0.01_pattern_9"   # 生成圖片資料夾
    gt_path = r"C:\Users\ericw\Desktop\bmp2\GT3\GT.bmp" # 單一 GT 圖片
    CSV_dir = r"C:\Users\ericw\Desktop\eval_result"

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = GenToOneGTDataset(gen_root, gt_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    test_generated_images(dataloader, device, CSV_dir)

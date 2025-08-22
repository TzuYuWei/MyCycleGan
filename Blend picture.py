import os
import torch
import lpips
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import vgg19, VGG19_Weights
from piq import ssim, psnr  # pip install piq
from torchvision.utils import save_image

# === Dataset (舊版算法對應) ===
class RainToGTDataset(Dataset):
    def __init__(self, gen_root, gt_root, transform=None):
        self.gen_paths = []
        self.gt_paths = []
        self.transform = transform

        gen_files = sorted([f for f in os.listdir(gen_root) if f.endswith('.png')])
        for gen_name in gen_files:
            # 對應 GT 檔名（去掉 _rain_ 或其他後綴）
            if "_rain_" in gen_name:
                base_name = gen_name.split("_rain_")[0] + ".png"
            else:
                base_name = gen_name.split("_leftImg8bit")[0] + "_leftImg8bit.png"

            gen_path = os.path.join(gen_root, gen_name)
            gt_path = os.path.join(gt_root, base_name)

            if os.path.exists(gt_path):
                self.gen_paths.append(gen_path)
                self.gt_paths.append(gt_path)
            else:
                print(f"❌ 無法配對 GT：{gt_path}")

    def __len__(self):
        return len(self.gen_paths)

    def __getitem__(self, idx):
        gen_img = Image.open(self.gen_paths[idx]).convert("RGB")
        gt_img = Image.open(self.gt_paths[idx]).convert("RGB")
        if self.transform:
            gen_img = self.transform(gen_img)
            gt_img = self.transform(gt_img)
        name = os.path.basename(self.gen_paths[idx])
        gt_name = os.path.basename(self.gt_paths[idx])
        return gen_img, gt_img, name, gt_name


# === Perceptual Loss (舊版算法) ===
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


# === 評估函數（舊版算法，已生成圖版） ===
def evaluate_generated_images(gen_dir, gt_dir, device="cuda", txt_out="metrics_results.txt"):
    transform = transforms.Compose([transforms.ToTensor()])

    dataset = RainToGTDataset(gen_dir, gt_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    lpips_fn = lpips.LPIPS(net='alex').to(device)
    perceptual_loss_fn = VGGPerceptualLoss().to(device)

    total_ssim = total_psnr = total_lpips = total_pl = total_edge_iou = 0
    count = 0

    out_dir = os.path.dirname(txt_out)
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)

    with open(txt_out, "w") as log_file, torch.no_grad():
        for gen_img, gt_img, name, gt_name in dataloader:
            gen_img, gt_img = gen_img.to(device), gt_img.to(device)

            # resize to 256x256 與舊版算法一致
            gt_img = F.interpolate(gt_img, size=(256, 256), mode='bilinear', align_corners=False)

            gen_img = gen_img.clamp(0.0, 1.0)
            gt_img = gt_img.clamp(0.0, 1.0)

            ssim_val = ssim(gen_img, gt_img, data_range=1.0).item()
            psnr_val = psnr(gen_img, gt_img, data_range=1.0).item()
            lpips_val = lpips_fn(gen_img, gt_img).mean().item()
            pl_val = perceptual_loss_fn(gen_img, gt_img).item()
            edge_val = edge_iou_opencv(gt_img, gen_img)

            total_ssim += ssim_val
            total_psnr += psnr_val
            total_lpips += lpips_val
            total_pl += pl_val
            total_edge_iou += edge_val
            count += 1

            print(f"{name} vs {gt_name} | SSIM: {ssim_val:.4f}, PSNR: {psnr_val:.2f}, "
                  f"LPIPS: {lpips_val:.4f}, PL: {pl_val:.4f}, EdgeIoU: {edge_val:.4f}")

            log_file.write(f"{name}, SSIM: {ssim_val:.4f}, PSNR: {psnr_val:.2f}, "
                           f"LPIPS: {lpips_val:.4f}, PL: {pl_val:.4f}, EDGE IoU: {edge_val:.4f}\n")

        log_file.write("\n=== AVERAGE ===\n")
        log_file.write(f"SSIM: {total_ssim/count:.4f}, PSNR: {total_psnr/count:.2f}, "
                       f"LPIPS: {total_lpips/count:.4f}, PL: {total_pl/count:.4f}, "
                       f"EDGE IoU: {total_edge_iou/count:.4f}\n")

    print("\n✅ 完成！平均指標：")
    print(f"SSIM: {total_ssim/count:.4f}, PSNR: {total_psnr/count:.2f}, "
          f"LPIPS: {total_lpips/count:.4f}, PL: {total_pl/count:.4f}, "
          f"EDGE IoU: {total_edge_iou/count:.4f}")


# === 執行範例 ===
if __name__ == "__main__":
    device = "cuda"
    gen_dir = r"C:\Users\ericw\Desktop\123"   # 已生成圖片資料夾
    gt_dir = r"C:\Users\ericw\Desktop\GT"     # GT 資料夾
    txt_out = r"C:\Users\ericw\Desktop\metrics_results.txt"

    evaluate_generated_images(gen_dir, gt_dir, device=device, txt_out=txt_out)

import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import vgg19, VGG19_Weights
from PIL import Image
import numpy as np
import cv2
import lpips

device = "cuda" if torch.cuda.is_available() else "cpu"

# ====== 資料夾 ======
lpips_dir = r"C:\Users\ericw\Desktop\picture_compare5\CycleGAN\LPIPS"
pl_dir = r"C:\Users\ericw\Desktop\picture_compare5\CycleGAN\PL"
gt_dir = r"C:\Users\ericw\Desktop\GT"

output_lpips_dir = lpips_dir + "_unqualified"
output_pl_dir = pl_dir + "_unqualified"

os.makedirs(output_lpips_dir, exist_ok=True)
os.makedirs(output_pl_dir, exist_ok=True)

# ====== LPIPS 模型 ======
lpips_fn = lpips.LPIPS(net='alex').to(device)

# ====== PL 模型 ======
class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT).features[:9]
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.eval()

    def forward(self, x, y):
        return F.l1_loss(self.vgg(x), self.vgg(y), reduction='none').mean(1, keepdim=True)

pl_fn = VGGPerceptualLoss().to(device)

# ====== 轉 tensor ======
transform = transforms.Compose([transforms.ToTensor()])

# ====== Patch 切分 ======
patch_size = 32  
lpips_threshold = 0.03
pl_threshold = 0.15

def get_patches(tensor_img, patch_size):
    _, _, H, W = tensor_img.shape
    patches = []
    positions = []
    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            h_end = min(i+patch_size, H)
            w_end = min(j+patch_size, W)
            patches.append(tensor_img[:,:,i:h_end,j:w_end])
            positions.append((i, h_end, j, w_end))
    return patches, positions

def compute_patch_scores(img_tensor, ref_tensor, map_type="LPIPS"):
    patches, positions = get_patches(img_tensor, patch_size)
    scores = []
    for idx, patch in enumerate(patches):
        i_start, i_end, j_start, j_end = positions[idx]
        ref_patch = ref_tensor[:, :, i_start:i_end, j_start:j_end]
        with torch.no_grad():
            if map_type=="LPIPS":
                dist = lpips_fn(patch, ref_patch)
                score = dist.item()
            elif map_type=="PL":
                pl_map = pl_fn(patch, ref_patch)
                score = pl_map.mean().item()
        scores.append(score)
    return scores, positions

def mark_unqualified_patches(img_path, gt_dir, map_type="LPIPS", threshold=0.05):
    fname = os.path.basename(img_path)
    # 用前3段匹配 GT
    prefix = "_".join(fname.split("_")[:3])
    gt_match = None
    for gt_fname in os.listdir(gt_dir):
        if gt_fname.startswith(prefix):
            gt_match = os.path.join(gt_dir, gt_fname)
            break
    if gt_match is None:
        print(f"No GT match for {fname}")
        return None, 0

    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    gt_img = Image.open(gt_match).convert("RGB")
    ref_tensor = transform(gt_img).unsqueeze(0).to(device)

    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    scores, positions = compute_patch_scores(img_tensor, ref_tensor, map_type)

    # 標記超過閾值的不合格 patch
    unqualified_count = 0
    for idx, score in enumerate(scores):
        if score > threshold:
            i_start, i_end, j_start, j_end = positions[idx]
            cv2.rectangle(img_cv, (j_start, i_start), (j_end, i_end), (0,0,255), 1)
            unqualified_count += 1

    return img_cv, unqualified_count

# ====== 執行 ======
def process_folder(input_dir, output_dir, map_type="LPIPS", threshold=0.05):
    stats = {}
    for fname in os.listdir(input_dir):
        if fname.lower().endswith((".png",".jpg",".jpeg")):
            input_path = os.path.join(input_dir, fname)
            marked_img, unqualified_count = mark_unqualified_patches(input_path, gt_dir, map_type, threshold)
            if marked_img is not None:
                cv2.imwrite(os.path.join(output_dir, fname), marked_img)
                stats[fname] = unqualified_count
    return stats

# LPIPS
lpips_stats = process_folder(lpips_dir, output_lpips_dir, map_type="LPIPS", threshold=lpips_threshold)
# PL
pl_stats = process_folder(pl_dir, output_pl_dir, map_type="PL", threshold=pl_threshold)

print("不合格 patch 標記完成！")
print("LPIPS 不合格數量：", lpips_stats)
print("PL 不合格數量：", pl_stats)

import cv2
import numpy as np
import os

# 資料夾路徑
gen_dir = r"C:\Users\ericw\Desktop\T\O\munster_000121_000019_leftImg8bit_rain_alpha_0.01_beta_0.005_dropsize_0.01_pattern_9"
orig_dir = r"C:\Users\ericw\Desktop\T\O\GT5"
output_dir = r"C:\Users\ericw\Desktop\T\A\munster_000121_000019_leftImg8bit_rain_alpha_0.01_beta_0.005_dropsize_0.01_pattern_9"
os.makedirs(output_dir, exist_ok=True)

def half_circle_blend_GT_core(gen_img, orig_img, cx, cy, r_core, r_blend):
    """
    在 (cx, cy) 為圓心的下半圓區域做 alpha blending
    - r_core: 核心半徑 (完全 GT)
    - r_blend: 漸變區厚度
    """
    h, w, c = gen_img.shape
    result = gen_img.copy()
    mask = np.zeros((h, w), dtype=np.float32)

    for y in range(h):
        for x in range(w):
            if y >= cy:  # 只取下半部分
                d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                if d <= r_core:
                    mask[y, x] = 1.0  # 完全 GT
                elif d <= r_core + r_blend:
                    mask[y, x] = 1 - (d - r_core) / r_blend  # 線性漸變
                # 其餘保持 0 (完全 GAN)

    mask = np.expand_dims(mask, axis=2)  # (h, w, 1)
    result = mask * orig_img.astype(np.float32) + (1 - mask) * gen_img.astype(np.float32)
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result




# 讀取唯一的 GT 圖
gt_files = [f for f in os.listdir(orig_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
if not gt_files:
    raise FileNotFoundError("⚠️ GT 資料夾裡沒有圖片")
gt_path = os.path.join(orig_dir, gt_files[0])
orig_img = cv2.imread(gt_path)
if orig_img is None:
    raise ValueError(f"⚠️ 讀取 GT 失敗: {gt_path}")

# 批次處理生成圖
for fname in os.listdir(gen_dir):
    if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
        continue

    gen_path = os.path.join(gen_dir, fname)
    gen_img = cv2.imread(gen_path)

    if gen_img is None:
        print(f"⚠️ 圖片讀取失敗: {fname}")
        continue

    # 做 blending
    result = half_circle_blend_GT_core(gen_img, orig_img, cx=138, cy=0, r_core=32, r_blend=64)

    # 存檔
    out_path = os.path.join(output_dir, fname)
    cv2.imwrite(out_path, result)

print("✅ 處理完成，結果已存到:", output_dir)

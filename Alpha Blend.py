import cv2
import numpy as np
import os

# 資料夾路徑（請換成你的資料夾路徑）
gen_dir = r"C:\Users\ericw\Desktop\SECBAM\result\origin"
orig_dir = r"C:\Users\ericw\Desktop\GT"
output_dir = r"C:\Users\ericw\Desktop\alpha blend3"
os.makedirs(output_dir, exist_ok=True)

import numpy as np
import cv2

def quarter_circle_blend_GT_core(gen_img, orig_img, r_core=32, r_blend=64):
    h, w, c = gen_img.shape
    result = gen_img.copy()  # 預設全部生成圖
    mask = np.zeros((h, w), dtype=np.float32)

    cx, cy = w-1, 0  # 右上角圓心

    for y in range(h):
        for x in range(w):
            # 只處理右上象限
            if x <= cx and y >= cy:
                d = np.sqrt((x - cx)**2 + (y - cy)**2)
                if d <= r_core:
                    mask[y, x] = 1.0  # 核心完全用原圖
                elif d <= r_core + r_blend:
                    mask[y, x] = 1 - (d - r_core) / r_blend  # 漸變
                # d > r_core + r_blend → mask[y,x] = 0, 完全生成圖

    mask = np.expand_dims(mask, axis=2)
    result = mask * orig_img.astype(np.float32) + (1 - mask) * gen_img.astype(np.float32)
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result

for fname in os.listdir(gen_dir):
    gen_path = os.path.join(gen_dir, fname)

    if "_rain_alpha" in fname:
        base_name = fname.split("_rain_alpha")[0] + ".png"
    else:
        continue

    orig_path = os.path.join(orig_dir, base_name)
    if not os.path.exists(orig_path):
        print(f"⚠️ 找不到對應的原圖: {base_name}")
        continue

    gen_img = cv2.imread(gen_path)
    orig_img = cv2.imread(orig_path)

    if gen_img is None or orig_img is None:
        print(f"⚠️ 圖片讀取失敗: {fname}")
        continue

    result = quarter_circle_blend_GT_core(gen_img, orig_img, r_core=32, r_blend=64)

    out_path = os.path.join(output_dir, fname)
    cv2.imwrite(out_path, result)

print("✅ 處理完成，結果已存到:", output_dir)

import cv2
import numpy as np
import os

# 資料夾路徑（請換成你的資料夾路徑）
gen_dir = r"C:\Users\ericw\Desktop\SECBAM\result\origin"
orig_dir = r"C:\Users\ericw\Desktop\GT"
output_dir = r"C:\Users\ericw\Desktop\alpha blend"
os.makedirs(output_dir, exist_ok=True)

def circular_corner_blend(gen_img, orig_img, r_core=8, r_blend=16):
    h, w, c = gen_img.shape
    mask = np.zeros((h, w), dtype=np.float32)

    # 右上角圓心
    cx, cy = w-1, 0  

    for y in range(h):
        for x in range(w):
            d = np.sqrt((x - cx)**2 + (y - cy)**2)

            if d <= r_core:
                mask[y, x] = 1.0
            elif d <= r_core + r_blend:
                mask[y, x] = 1 - (d - r_core) / r_blend
            else:
                mask[y, x] = 0.0

    mask = np.expand_dims(mask, axis=2)

    blended = (mask * orig_img.astype(np.float32) +
               (1 - mask) * gen_img.astype(np.float32))
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return blended

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

    result = circular_corner_blend(gen_img, orig_img, r_core=8, r_blend=16)

    out_path = os.path.join(output_dir, fname)
    cv2.imwrite(out_path, result)

print("✅ 處理完成，結果已存到:", output_dir)

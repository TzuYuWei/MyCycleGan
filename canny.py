import cv2
import os

# 原始圖片資料夾
input_dir = r"C:\Users\ericw\Desktop\picture_compare4\CycleGAN\EdgeIoU"

# 輸出資料夾
output_dir = r"C:\Users\ericw\Desktop\picture_compare4\CycleGAN\EdgeIoU_canny"
os.makedirs(output_dir, exist_ok=True)

# Canny 邊緣檢測參數
threshold1 = 100
threshold2 = 200

# 處理每張圖片
for fname in os.listdir(input_dir):
    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(input_dir, fname)
        
        # 讀灰階
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Canny 邊緣檢測
        edges = cv2.Canny(img, threshold1, threshold2)
        
        # 存檔
        save_path = os.path.join(output_dir, fname)
        cv2.imwrite(save_path, edges)

print("完成 Canny 邊緣處理！")
# test_main.py
# ✅ 載入訓練好的模型，配對測試資料集並計算評估指標

import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from test_metrics import RainToGTDataset, test_model
from CycleGAN import Generator  # 根據你使用的模型架構載入
from torchvision.transforms import InterpolationMode

if __name__ == "__main__":
    # === 裝置設定 ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 資料路徑設定 ===
    rain_root = r'C:\Users\ericw\Desktop\testA'
    gt_root = r'C:\Users\ericw\Desktop\testB'
    model_path = r'C:\Users\ericw\Desktop\CycleGAN_ALL\models\checkpoint_epoch100.pth'
    save_dir = r'C:\Users\ericw\Desktop\CycleGAN_ALL\result'
    TXT_dir = r'C:\Users\ericw\Desktop\CycleGAN_ALL\result\train_mean'

    # === 圖片轉換設定 ===
    transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])

    # === 初始化 Generator 並載入模型權重 ===
    generator = Generator().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    generator.load_state_dict(checkpoint["generator_A2B"])

    # === 逐城市資料夾測試（含 flip）===
    for city in os.listdir(rain_root):
        rain_city_path = os.path.join(rain_root, city)
        gt_city_path = os.path.join(gt_root, city)

        # 路徑必須都存在
        if not os.path.isdir(rain_city_path) or not os.path.isdir(gt_city_path):
            print(f"❌ 跳過 {city}：路徑不存在")
            continue

        print(f"\n✅ 測試中：{city}")
        print(f"Rain: {rain_city_path}")
        print(f"GT  : {gt_city_path}")

        test_dataset = RainToGTDataset(rain_city_path, gt_city_path, transform=transform)
        if len(test_dataset) == 0:
            print(f"⚠️ 資料夾 {city} 沒有有效圖片，跳過")
            continue

        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        test_model(generator, test_loader, device, save_dir, TXT_dir)




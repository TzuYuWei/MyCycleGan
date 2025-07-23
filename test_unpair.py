# test_main.py
# ✅ 載入訓練好的模型，配對測試資料集並計算評估指標

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from test_metrics import RainToGTDataset, test_model
from CycleGANSE import Generator  # 根據你使用的模型架構載入
from torchvision.transforms import InterpolationMode

if __name__ == "__main__":
    # === 裝置設定 ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 資料路徑設定 ===
    rain_root = r'C:\Users\User\Desktop\CycleGAN+SE_128\test\testA'
    gt_root = r'C:\Users\User\Desktop\CycleGAN+SE_128\test\testB'
    model_path = r'C:\Users\User\Desktop\CycleGAN+SE_128\models\checkpoint_epoch100.pth'
    save_dir = r'C:\Users\User\Desktop\CycleGAN+SE_128\result\test result_128'
    TXT_dir = r'C:\Users\User\Desktop\CycleGAN+SE_128\result\train_mean'

    # === 圖片轉換設定（與訓練一致） ===
    transform = transforms.Compose([
        transforms.Resize((128, 128), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])

    # === 載入測試資料集 ===
    test_dataset = RainToGTDataset(rain_root, gt_root, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # === 初始化 Generator 並載入模型權重 ===
    generator = Generator().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    generator.load_state_dict(checkpoint["generator_A2B"])

    # === 執行測試與評估 ===
    test_model(generator, test_loader, device, save_dir, TXT_dir)

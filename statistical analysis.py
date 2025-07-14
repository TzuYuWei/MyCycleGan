import re
import numpy as np
import pandas as pd

# 讀取 TXT 檔案
txt_path = r'C:\Users\ericw\Desktop\口試後實驗數據\rain_SE_freq_128\train_SE_freq_128.txt'
with open(txt_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 正規表示式提取數值
pattern = r'SSIM:\s*(\d+\.\d+),\s*PSNR:\s*(\d+\.\d+)\s*dB,\s*LPIPS:\s*(\d+\.\d+),\s*PL:\s*(\d+\.\d+),\s*EDGE IoU:\s*(\d+\.\d+),\s*mIoU:\s*(\d+\.\d+)'

X = []
for line in lines:
    match = re.search(pattern, line)
    if match:
        numbers = [float(x) for x in match.groups()]
        X.append(numbers)

# 如果找不到任何數值就提醒使用者
if not X:
    print("⚠️ 沒有成功從檔案中抓到任何指標數據，請確認格式是否一致。")
else:
    # 轉為 DataFrame 方便統計分析
    X = np.array(X)
    columns = ['SSIM', 'PSNR', 'LPIPS', 'PL', 'EDGE_IoU', 'mIoU']
    df = pd.DataFrame(X, columns=columns)

    # 顯示描述性統計
    print("📊 描述性統計：")
    print(df.describe())

    # Q3/Q1 穩定性指標
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)

    print("\n📈 Q3/Q1（穩定性指標）：")
    for col in df.columns:
        if q1[col] != 0:
            stability = q3[col] / q1[col]
            print(f"{col}: {stability:.4f}")
        else:
            print(f"{col}: Q1 為 0，無法計算 Q3/Q1")

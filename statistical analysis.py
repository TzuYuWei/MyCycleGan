import re
import numpy as np
import pandas as pd

# === 讀取 TXT 檔案 ===
txt_path = r'C:\Users\ericw\Desktop\CycleGAN_flip_128\result\train_mean\test_results.txt'
with open(txt_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# === 正則表達式提取指標數值 ===
pattern = r'SSIM:\s*([\d.]+),\s*PSNR:\s*([\d.]+)\s*dB,\s*LPIPS:\s*([\d.]+),\s*PL:\s*([\d.]+),\s*EDGE IoU:\s*([\d.]+),\s*mIoU:\s*([\d.]+)'

X = []
for line in lines:
    match = re.search(pattern, line)
    if match:
        numbers = [float(x) for x in match.groups()]
        X.append(numbers)

# === 處理統計與輸出 CSV ===
if not X:
    print("⚠️ 沒有成功從檔案中抓到任何指標數據，請確認格式是否一致。")
else:
    # 建立 DataFrame
    X = np.array(X)
    columns = ['SSIM', 'PSNR', 'LPIPS', 'PL', 'EDGE_IoU', 'mIoU']
    df = pd.DataFrame(X, columns=columns)

    # 描述性統計
    desc_stats = df.describe()

    # Q3/Q1 穩定性
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    stability_data = {}
    for col in df.columns:
        if q1[col] != 0:
            stability_data[col] = q3[col] / q1[col]
        else:
            stability_data[col] = np.nan  # 避免除以 0

    # 將統計數據與穩定性結果合併到同一份 CSV
    result_df = desc_stats.copy()
    result_df.loc['Q3/Q1'] = pd.Series(stability_data)

    # 儲存 CSV 檔案
    output_path = r'C:\Users\ericw\Desktop\CycleGAN_flip_128\result\train_mean\stat_results.csv'  # <<<←←←← 在這裡填入實際儲存路徑
    result_df.to_csv(output_path, encoding='utf-8-sig')
    print(f"✅ 統計結果已儲存至：{output_path}")

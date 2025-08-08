import re
import numpy as np
import pandas as pd
import os

# === 檔案路徑 ===
base_dir = r'C:\Users\ericw\Desktop\CycleGAN_SE_CBAM_ALL\result\test_mean'
origin_txt = os.path.join(base_dir, 'test_results_origin.txt')
flip_txt = os.path.join(base_dir, 'test_results_flip.txt')

# === 正則表達式模式 ===
pattern = r'SSIM:\s*([\d.]+),\s*PSNR:\s*([\d.]+)\s*dB,\s*LPIPS:\s*([\d.]+),\s*PL:\s*([\d.]+),\s*EDGE IoU:\s*([\d.]+),\s*mIoU:\s*([\d.]+)'

def parse_txt(txt_path):
    data = []
    if not os.path.exists(txt_path):
        print(f"⚠️ 找不到檔案：{txt_path}")
        return pd.DataFrame()  # 回傳空的 DataFrame
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        match = re.search(pattern, line)
        if match:
            numbers = [float(x) for x in match.groups()]
            data.append(numbers)
    if not data:
        print(f"⚠️ 沒有從 {txt_path} 解析到數據，請確認格式")
        return pd.DataFrame()
    df = pd.DataFrame(data, columns=['SSIM', 'PSNR', 'LPIPS', 'PL', 'EDGE_IoU', 'mIoU'])
    return df

# === 分別解析 ===
df_origin = parse_txt(origin_txt)
df_flip = parse_txt(flip_txt)

# === 合併統計 ===
combined_df = pd.concat([df_origin, df_flip], ignore_index=True)

# === 統計函式 ===
def add_stats(df, name):
    if df.empty:
        print(f"⚠️ {name} 資料為空，略過統計")
        return pd.DataFrame()
    desc_stats = df.describe()
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    stability = {}
    for col in df.columns:
        if q1[col] != 0:
            stability[col] = q3[col] / q1[col]
        else:
            stability[col] = np.nan
    desc_stats.loc['Q3/Q1'] = pd.Series(stability)
    return desc_stats

# === 建立統計結果 ===
origin_stats = add_stats(df_origin, 'origin')
flip_stats = add_stats(df_flip, 'flip')
combined_stats = add_stats(combined_df, 'combined')

# === 儲存 CSV ===
if not origin_stats.empty:
    origin_stats.to_csv(os.path.join(base_dir, 'stat_results_origin.csv'), encoding='utf-8-sig')
    print("✅ 已儲存 stat_results_origin.csv")
if not flip_stats.empty:
    flip_stats.to_csv(os.path.join(base_dir, 'stat_results_flip.csv'), encoding='utf-8-sig')
    print("✅ 已儲存 stat_results_flip.csv")
if not combined_stats.empty:
    combined_stats.to_csv(os.path.join(base_dir, 'stat_results_combined.csv'), encoding='utf-8-sig')
    print("✅ 已儲存 stat_results_combined.csv")

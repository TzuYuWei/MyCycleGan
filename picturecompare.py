import os
import re
import shutil
import pandas as pd

# ====== 可調整參數 ======
txt1 = r"C:\Users\ericw\Desktop\results\test_results_SECBAM.txt"
txt2 = r"C:\Users\ericw\Desktop\results\test_results_CycleGAN.txt"
img_dir1 = r"C:\Users\ericw\Desktop\alpha blend"
img_dir2 = r"C:\Users\ericw\Desktop\口試後雨天實驗數據\CycleGAN_ALL\result\origin"
model1_name = r"C:\Users\ericw\Desktop\picture_compare5\SECBAM"
model2_name = r"C:\Users\ericw\Desktop\picture_compare5\CycleGAN"
base_output_dir = r"C:\Users\ericw\Desktop\picture_compare5"

# 閾值
LPIPS_thr = 0.02
PL_thr = 0.01
EDGE_thr = 0.1   # EdgeIoU 還是保留原本
LPIPS_small = 0.01
PL_small = 0.005

TOP_N = 150   # 取前 N 張 (0 = 全部)
# ======================

def parse_txt(txt_file):
    results = {}
    with open(txt_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            match = re.match(
                r"([^,]+), SSIM: ([0-9.]+), PSNR: ([0-9.]+) dB, "
                r"LPIPS: ([0-9.]+), PL: ([0-9.]+), EDGE IoU: ([0-9.]+)", line)
            if match:
                fname, ssim, psnr, lpips, pl, edge = match.groups()
                results[fname] = {
                    "SSIM": float(ssim),
                    "PSNR": float(psnr),
                    "LPIPS": float(lpips),
                    "PL": float(pl),
                    "EDGEIOU": float(edge),
                }
    return results

# 讀取兩個模型的 txt
data1 = parse_txt(txt1)
data2 = parse_txt(txt2)

# 建立輸出資料夾
for m in [model1_name, model2_name]:
    for sub in ["EdgeIoU", "LPIPS", "PL"]:
        os.makedirs(os.path.join(base_output_dir, m, sub), exist_ok=True)

# 用來存記錄
records_LPIPS, records_PL, records_EDGE = [], [], []

for fname in set(data1.keys()) & set(data2.keys()):
    m1 = data1[fname]
    m2 = data2[fname]

    # --------- LPIPS 情境（忽略 EDGE） ----------
    if abs(m1["LPIPS"] - m2["LPIPS"]) > LPIPS_thr:
        diff_val = abs(m1["LPIPS"] - m2["LPIPS"])
        records_LPIPS.append({
            "filename": fname,
            "diff_val": diff_val,
            f"SSIM_{model1_name}": m1["SSIM"], f"SSIM_{model2_name}": m2["SSIM"],
            f"PSNR_{model1_name}": m1["PSNR"], f"PSNR_{model2_name}": m2["PSNR"],
            f"LPIPS_{model1_name}": m1["LPIPS"], f"LPIPS_{model2_name}": m2["LPIPS"],
            f"PL_{model1_name}": m1["PL"], f"PL_{model2_name}": m2["PL"],
            f"EDGEIOU_{model1_name}": m1["EDGEIOU"], f"EDGEIOU_{model2_name}": m2["EDGEIOU"],
        })

    # --------- PL 情境（忽略 EDGE） ----------
    if abs(m1["PL"] - m2["PL"]) > PL_thr:
        diff_val = abs(m1["PL"] - m2["PL"])
        records_PL.append({
            "filename": fname,
            "diff_val": diff_val,
            f"SSIM_{model1_name}": m1["SSIM"], f"SSIM_{model2_name}": m2["SSIM"],
            f"PSNR_{model1_name}": m1["PSNR"], f"PSNR_{model2_name}": m2["PSNR"],
            f"LPIPS_{model1_name}": m1["LPIPS"], f"LPIPS_{model2_name}": m2["LPIPS"],
            f"PL_{model1_name}": m1["PL"], f"PL_{model2_name}": m2["PL"],
            f"EDGEIOU_{model1_name}": m1["EDGEIOU"], f"EDGEIOU_{model2_name}": m2["EDGEIOU"],
        })

    # --------- EdgeIoU 情境（保留原本） ----------
    if abs(m1["EDGEIOU"] - m2["EDGEIOU"]) > EDGE_thr and \
       abs(m1["LPIPS"] - m2["LPIPS"]) < LPIPS_small and abs(m1["PL"] - m2["PL"]) < PL_small:
        diff_val = abs(m1["EDGEIOU"] - m2["EDGEIOU"])
        records_EDGE.append({
            "filename": fname,
            "diff_val": diff_val,
            f"SSIM_{model1_name}": m1["SSIM"], f"SSIM_{model2_name}": m2["SSIM"],
            f"PSNR_{model1_name}": m1["PSNR"], f"PSNR_{model2_name}": m2["PSNR"],
            f"LPIPS_{model1_name}": m1["LPIPS"], f"LPIPS_{model2_name}": m2["LPIPS"],
            f"PL_{model1_name}": m1["PL"], f"PL_{model2_name}": m2["PL"],
            f"EDGEIOU_{model1_name}": m1["EDGEIOU"], f"EDGEIOU_{model2_name}": m2["EDGEIOU"],
        })

# --------- 依差異排序 & 取 Top-N ---------
def select_top(records, top_n):
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df = df.sort_values("diff_val", ascending=False)
    if top_n > 0:
        df = df.head(top_n)
    return df

df_LPIPS = select_top(records_LPIPS, TOP_N)
df_PL = select_top(records_PL, TOP_N)
df_EDGE = select_top(records_EDGE, TOP_N)

# --------- 複製圖片 & 輸出 CSV ---------
def save_results(df, subdir, csv_name):
    if df.empty:
        return
    for _, row in df.iterrows():
        fname = row["filename"]
        for model_name, img_dir in [(model1_name, img_dir1), (model2_name, img_dir2)]:
            src = os.path.join(img_dir, fname)
            if os.path.exists(src):
                dst = os.path.join(base_output_dir, model_name, subdir, fname)
                shutil.copy(src, dst)
    df.drop(columns=["diff_val"]).to_csv(os.path.join(base_output_dir, model1_name, csv_name), index=False)
    df.drop(columns=["diff_val"]).to_csv(os.path.join(base_output_dir, model2_name, csv_name), index=False)

save_results(df_LPIPS, "LPIPS", "LPIPS.csv")
save_results(df_PL, "PL", "PL.csv")
save_results(df_EDGE, "EdgeIoU", "EdgeIoU.csv")

print(f"LPIPS 情境: 找到 {len(df_LPIPS)} 張 (Top-{TOP_N if TOP_N>0 else 'All'})")
print(f"PL 情境: 找到 {len(df_PL)} 張 (Top-{TOP_N if TOP_N>0 else 'All'})")
print(f"EdgeIoU 情境: 找到 {len(df_EDGE)} 張 (Top-{TOP_N if TOP_N>0 else 'All'})")


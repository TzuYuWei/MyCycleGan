import csv
import re
from collections import defaultdict

metrics = {
    'SSIM': 'higher',
    'PSNR': 'higher',
    'LPIPS': 'lower',
    'PL': 'lower',
    'EDGE IoU': 'higher',
    'mIoU': 'higher',
}

def parse_metrics(file_path):
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            match = re.match(
                r'([^,]+), SSIM: ([\d.]+), PSNR: ([\d.]+) dB, LPIPS: ([\d.]+), PL: ([\d.]+), EDGE IoU: ([\d.]+), mIoU: ([\d.]+)',
                line.strip()
            )
            if match:
                filename = match.group(1).strip()
                values = {
                    'SSIM': float(match.group(2)),
                    'PSNR': float(match.group(3)),
                    'LPIPS': float(match.group(4)),
                    'PL': float(match.group(5)),
                    'EDGE IoU': float(match.group(6)),
                    'mIoU': float(match.group(7)),
                }
                data[filename] = values
    return data

def extract_spec_keys(filename):
    specs = []
    pattern_match = re.search(r'pattern_\d+', filename)
    alpha_match = re.search(r'alpha_\d+\.\d+', filename)
    if pattern_match:
        specs.append(pattern_match.group())
    if alpha_match:
        specs.append(alpha_match.group())
    return specs

def compare_by_specs(file1, file2, output_csv, method1, method2):
    data1 = parse_metrics(file1)
    data2 = parse_metrics(file2)

    all_filenames = set(data1.keys()) & set(data2.keys())

    stats = defaultdict(lambda: {
        metric: {method1: 0, method2: 0}
        for metric in metrics
    })

    for filename in all_filenames:
        specs = extract_spec_keys(filename)
        for spec in specs:
            for metric, rule in metrics.items():
                v1 = data1[filename][metric]
                v2 = data2[filename][metric]

                if (rule == 'higher' and v1 > v2) or (rule == 'lower' and v1 < v2):
                    stats[spec][metric][method1] += 1
                else:
                    stats[spec][metric][method2] += 1

    # 排序 alpha 在前、pattern 在後
    def sort_key(spec):
        if 'alpha' in spec:
            return (0, float(spec.split('_')[1]))
        elif 'pattern' in spec:
            return (1, int(spec.split('_')[1]))
        else:
            return (2, spec)

    sorted_specs = sorted(stats.keys(), key=sort_key)

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['規格名'] + list(metrics.keys()) + ['比值']
        writer.writerow(header)

        for spec in sorted_specs:
            row = [spec]
            total_method1 = 0
            total_method2 = 0

            for metric in metrics:
                count1 = stats[spec][metric][method1]
                count2 = stats[spec][metric][method2]
                total_method1 += count1
                total_method2 += count2
                row.append(f"{method1}({count1}) {method2}({count2})")

            total = total_method1 + total_method2
            if total == 0:
                ratio_str = "0% : 0%"
            else:
                pct1 = round(total_method1 / total * 100)
                pct2 = 100 - pct1
                ratio_str = f"{pct1}% : {pct2}%"

            row.append(ratio_str)
            writer.writerow(row)

    print(f'最終比較結果已寫入：{output_csv}')

# === 執行區 ===
file1 = r'C:\Users\ericw\Desktop\口試後實驗數據\CycleGAN_FCA_ALL\result\train_mean\test_results_origin.txt'      # 方法1的結果
file2 = r'C:\Users\ericw\Desktop\口試後實驗數據\CycleGAN_CBAM_ALL\result\test_mean\test_results_origin.txt'     # 方法2的結果
output_csv = r'C:\Users\ericw\Desktop\口試後實驗數據\FCAVSCBAM_ALL_compare.csv'

# 自訂方法名稱
method1 = 'FCA'
method2 = 'CBAM'

compare_by_specs(file1, file2, output_csv, method1=method1, method2=method2)
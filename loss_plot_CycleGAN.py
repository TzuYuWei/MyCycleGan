import matplotlib.pyplot as plt

# 讀取資料
epochs = []
loss_g = []
loss_d_a = []
loss_d_b = []

with open(r'C:\Users\ericw\Desktop\CycleGAN_FCA_SE_ALL\loss_plot\train_loss_log.csv', "r") as f:
    for line in f:
        if line.strip() == "":
            continue  # 跳過空行
        parts = line.strip().split(',')  # ✅ 改為逗號分隔
        if len(parts) != 4:
            print("格式錯誤行：", line.strip())
            continue
        try:
            epochs.append(int(parts[0]))
            loss_g.append(float(parts[1]))
            loss_d_a.append(float(parts[2]))
            loss_d_b.append(float(parts[3]))
        except ValueError:
            print("轉換錯誤行：", parts)
            continue

print("✅ 前 5 筆 Loss_G：", loss_g[:5])
print("✅ 前 5 筆 Epochs：", epochs[:5])

# 畫圖
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss_g, label="Generator Loss (G)")
plt.plot(epochs, loss_d_a, label="Discriminator A Loss (D_A)")
plt.plot(epochs, loss_d_b, label="Discriminator B Loss (D_B)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(r"C:\Users\ericw\Desktop\CycleGAN_FCA_SE_ALL\loss_plot\loss_curve.png")
plt.show()

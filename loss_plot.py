import pandas as pd
import matplotlib.pyplot as plt

# 讀取 CSV 檔案
file_path = r'C:\Users\ericw\Desktop\loss_log_128.csv'
df = pd.read_csv(file_path)

# 找出 Validation Loss 最小的 epoch
min_val_loss = df['val_loss'].min()
best_epoch = df.loc[df['val_loss'] == min_val_loss, 'epoch'].values[0]
print(f"Validation Loss 最佳模型在 epoch {best_epoch}，val_loss = {min_val_loss:.4f}")

# 畫出 Train Loss 和 Validation Loss 曲線 + 最佳 epoch 標註
plt.figure(figsize=(10, 6))
plt.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', marker='o')

# 標註最佳 Validation Loss 點
plt.scatter(best_epoch, min_val_loss, color='red', label=f'Best Epoch ({best_epoch})')
plt.text(best_epoch, min_val_loss + 0.05, f'{min_val_loss:.3f}', color='red')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from origin_metric import GenToGTDataset, test_generated_images

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gen_root = r"C:\Users\ericw\Desktop\alpha blend3"   # 你已經產生好的圖
    gt_root = r"C:\Users\ericw\Desktop\testB"               # GT 圖片
    TXT_dir = r"C:\Users\ericw\Desktop\results"

    transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])

    dataset = GenToGTDataset(gen_root, gt_root, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # quantize=True → 和舊程式的數字幾乎完全一致
    test_generated_images(loader, device, TXT_dir, quantize=False)



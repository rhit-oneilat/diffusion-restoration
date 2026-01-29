import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import glob
import os
from PIL import Image
from torch.utils.data import Dataset

IMG_PATH = "/work/csse463/Ours/data/celeba_raw/img_align_celeba"

class SimpleDataset(Dataset):
    def __init__(self, root):
        self.files = glob.glob(os.path.join(root, "*.jpg"))[:100] # Check first 100
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    def __len__(self): return len(self.files)
    def __getitem__(self, i): return self.transform(Image.open(self.files[i]).convert("RGB"))

def check():
    ds = SimpleDataset(IMG_PATH)
    loader = DataLoader(ds, batch_size=32)
    batch = next(iter(loader))

    print(f"Batch Shape: {batch.shape}")
    print(f"Min Value: {batch.min().item():.4f} (Should be near -1.0)")
    print(f"Max Value: {batch.max().item():.4f} (Should be near 1.0)")

    if batch.min() > -0.1:
        print("FAIL: Data is not normalized to [-1, 1]. It looks like [0, 1].")
    elif batch.max() > 10:
        print("FAIL: Data looks like [0, 255]. Add ToTensor()!")
    else:
        print("Data range looks correct.")

if __name__ == "__main__":
    check()

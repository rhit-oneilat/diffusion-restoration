import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import glob
import os

# --- CONFIG ---
# Update this to match where you unzipped the images
# It should point to the folder containing the .jpg files directly
IMAGE_FOLDER_PATH = "/home/oneilat/463/data/celeba_raw/img_align_celeba/" 
IMG_SIZE = 64
BATCH_SIZE = 32

print(f"--- Checking for images in {IMAGE_FOLDER_PATH} ---")

# --- 1. ROBUST CUSTOM LOADER ---
class SimpleFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # Finds all jpgs in the folder, ignores strict structure
        self.files = glob.glob(os.path.join(root_dir, "*.jpg"))
        self.transform = transform
        if len(self.files) == 0:
            raise ValueError(f"No images found in {root_dir}. Check your path!")
        # Slice to 1000 for speed
        self.files = self.files[:1000]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.files[idx]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, 0 # Return dummy label
        except Exception as e:
            print(f"Error loading {self.files[idx]}: {e}")
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), 0

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = SimpleFolderDataset(IMAGE_FOLDER_PATH, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print(f"Successfully loaded {len(dataset)} images.")

print("\n--- Step 2: Calculating Baseline ---")
clean, _ = next(iter(loader))
noise = torch.randn_like(clean) * 0.1
noisy = clean + noise
baseline_mse = nn.functional.mse_loss(noisy, clean)
print(f"BASELINE MSE (Target to beat): {baseline_mse.item():.6f}")

print("\n--- Step 3: Training Simple CNN ---")
model = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
    nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), # Added one layer for better stability
    nn.Conv2d(32, 3, 3, padding=1)
)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(3):
    total_loss = 0
    for x, _ in loader:
        noise = torch.randn_like(x) * 0.1
        noisy_input = x + noise
        
        optimizer.zero_grad()
        output = model(noisy_input)
        loss = criterion(output, x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss/len(loader):.6f}")

print("\n--- DONE. Record these numbers. ---")

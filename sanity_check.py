import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.unet import UNet
from src.diffusion import DiffusionModule
from torchvision import transforms
from PIL import Image
import glob
import os

# CONFIG
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_PATH = "/work/csse463/Ours/data/celeba_raw/img_align_celeba"

def sanity_check():
    print(f"--- STARTING OVERFIT TEST on {DEVICE} ---")

    files = glob.glob(os.path.join(IMG_PATH, "*.jpg"))
    if not files:
        print("Error: No images found. Check path.")
        return

    img = Image.open(files[0]).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Map to [-1, 1]
    ])
    x = transform(img).unsqueeze(0).to(DEVICE)

    batch_size = 8
    batch = x.repeat(batch_size, 1, 1, 1)

    model = UNet().to(DEVICE)
    diffusion = DiffusionModule().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3) # High LR for fast overfit
    mse = torch.nn.MSELoss()

    print("Step 0: Initializing Loop...")

    model.train()
    for step in range(301):
        optimizer.zero_grad()

        # Sample t
        t = diffusion.sample_timesteps(batch_size).to(DEVICE)

        # Forward Noise
        x_t, noise = diffusion.noise_images(batch, t)

        # Predict Noise
        predicted_noise = model(x_t, t)

        # Loss
        loss = mse(noise, predicted_noise)
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"Step {step}: Loss = {loss.item():.6f}")

    if loss.item() < 0.005:
        print("\nSUCCESS: Model successfully overfit a single batch.")
        print("    Your U-Net connections and Gradient flow are working.")
    else:
        print("\nFAILURE: Loss did not converge.")
        print("    Check: Normalization, Skip Connections in U-Net, or Time Embedding shapes.")

if __name__ == "__main__":
    sanity_check()

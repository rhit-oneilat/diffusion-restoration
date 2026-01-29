import torch
import torchvision
from src.diffusion import DiffusionModule
from PIL import Image
from torchvision import transforms
import glob
import os

# CONFIG
IMG_PATH = "/work/csse463/Ours/data/celeba_raw/img_align_celeba"

def visualize():
    device = "cpu" # CPU is fine for this

    files = glob.glob(os.path.join(IMG_PATH, "*.jpg"))
    img = Image.open(files[0]).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    x = transform(img).unsqueeze(0).to(device) # (1, 3, 64, 64)

    # Check t=0 (Start), t=250, t=500, t=750, t=999 (End)
    timesteps = torch.tensor([0, 250, 500, 750, 999]).long().to(device)

    diffusion = DiffusionModule().to(device)

    noisy_images = []

    noisy_images.append(x)

    for t_val in timesteps:
        t_tensor = torch.tensor([t_val]).long().to(device)
        x_t, _ = diffusion.noise_images(x, t_tensor)
        noisy_images.append(x_t)

    grid = torch.cat(noisy_images, dim=0)

    # Rescale from [-1, 1] back to [0, 1] for saving
    grid = (grid.clamp(-1, 1) + 1) / 2
    torchvision.utils.save_image(grid, "schedule_vis.png", nrow=6)
    print("Saved 'schedule_vis.png'. Download it and verify t=999 is pure noise.")

if __name__ == "__main__":
    visualize()

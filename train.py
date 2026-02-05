import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import logging
from src.unet import UNet
from src.diffusion import DiffusionModule
from src.CelebADataset import CelebADataset

# Setup logging
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S %p")

def train(args):
    device = args.device
    dataloader = get_data(args)
    model = UNet(device=device).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr) 
    mse = nn.MSELoss()
    diffusion = DiffusionModule(img_size=args.image_size, device=device)
    
    logging.info(f"Starting training on {device}...")
    
    for epoch in range(args.epochs):
        logging.info(f"Epoch {epoch+1}/{args.epochs}")
        pbar = tqdm(dataloader)
        epoch_loss = 0
        
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(MSE=loss.item())
        
        avg_loss = epoch_loss / len(dataloader)
        logging.info(f"Epoch {epoch+1} Loss: {avg_loss}")

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join("model.pt"))
            logging.info(f"Saved checkpoint to model.pt")

def get_data(args):
    transforms_list = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)), # 64x64
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # [-1, 1] range
    ])
    dataset = CelebADataset(args.dataset_path, transform=transforms_list)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader

class Args:
    pass

if __name__ == "__main__":
    args = Args()
    args.batch_size = 12 # maybe could increase bc of VRAM capacity
    args.image_size = 64
    args.dataset_path = "/work/csse463/202620/06/diffusion-restoration/data/celeba_raw/img_align_celeba"
    args.device = "cuda:6" if torch.cuda.is_available() else "cpu"
    args.lr = 3e-4
    args.epochs = 500 # Can be changed
    
    # Check if data path exists, otherwise warn
    if not os.path.exists("./data/celeba_raw/img_align_celeba"):
        logging.warning("Data directory not found. Please ensure images are in ./data/celeba_raw/img_align_celeba")
        # Creating a dummy directory for structure if needed or just letting it fail gracefully/warn
    
    train(args)

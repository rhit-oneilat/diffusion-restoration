import os
from src.diffusionInpainting import InpaintingDiffusion
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import logging
from src.unet import UNet
from src.CelebADataset import CelebADataset

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%I:%M:%S %p",
    handlers=[
        logging.FileHandler("train_log.txt"),
        logging.StreamHandler()
    ]
)

def train(args):
    device = args.device
    dataloader = get_data(args)
    model = UNet(c_in=4, device=device).to(device)
    #change device lines with args.device to match
    model = nn.DataParallel(model, device_ids=[1, 3, 6, 7])  
    optimizer = optim.AdamW(model.parameters(), lr=args.lr) 
    mse = nn.MSELoss()
    diffusion = InpaintingDiffusion(img_size=args.image_size, device=device)
    
    logging.info(f"Starting training on {device}...")
    
    for epoch in range(args.epochs):
        logging.info(f"Epoch {epoch+1}/{args.epochs}")
        pbar = tqdm(dataloader)
        epoch_loss = 0
        
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            # Create a random mask for inpainting conditioning (1 = known, 0 = unknown)
            mask = diffusion.create_random_mask(images.shape[0]).to(device)
            # Concatenate mask as an extra channel -> model expects 4 channels (RGB + mask)
            model_input = torch.cat([x_t, mask], dim=1)

            predicted_noise = model(model_input, t)
            loss = mse(noise, predicted_noise)

            # Skip and log if loss is NaN or Inf
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"Batch {i}: loss is NaN or Inf ({loss}). Skipping optimizer step.")
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()

            # Check gradients for NaN/Inf before applying updates
            grad_is_finite = True
            max_grad = 0.0
            for p in model.parameters():
                if p.grad is None:
                    continue
                g = p.grad
                if torch.isnan(g).any() or torch.isinf(g).any():
                    grad_is_finite = False
                    break
                try:
                    gmax = g.abs().max().item()
                except RuntimeError:
                    gmax = 0.0
                if gmax > max_grad:
                    max_grad = gmax

            if not grad_is_finite:
                logging.warning(f"Batch {i}: found NaN/Inf in gradients. Skipping optimizer step.")
                optimizer.zero_grad()
                continue

            # Gradient clipping to stabilize training (safety net for exploding gradients)
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(MSE=loss.item(), GradNorm=f"{total_norm:.4f}", MaxGrad=f"{max_grad:.4f}")
        
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
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2
    )
    return dataloader

class Args:
    pass

if __name__ == "__main__":
    args = Args()
    args.batch_size = 48  # Increased from 12 - 4 GPUs can handle much more
    args.image_size = 64
    args.dataset_path = "/work/csse463/202620/06/diffusion-restoration/data/celeba_raw/img_align_celeba"
    args.device = "cuda:1" if torch.cuda.is_available() else "cpu"
    args.lr = 3e-4
    args.epochs = 500 # Can be changed
    
    # Check if data path exists, otherwise warn
    if not os.path.exists("./data/celeba_raw/img_align_celeba"):
        logging.warning("Data directory not found. Please ensure images are in ./data/celeba_raw/img_align_celeba")
        # Creating a dummy directory for structure if needed or just letting it fail gracefully/warn
    
    train(args)

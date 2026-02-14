import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import logging
from src.unet import UNet
from src.CelebADataset import CelebADataset
from src.diffusionInpainting import InpaintingDiffusion

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
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    diffusion = InpaintingDiffusion(img_size=args.image_size, device=device)
    logging.info(f"Starting training on {device} with batch size {args.batch_size}...")

    for epoch in range(args.epochs):
        logging.info(f"Epoch {epoch+1}/{args.epochs}")
        model.train()
        pbar = tqdm(dataloader)
        epoch_loss = 0

        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)

            # FIX 1: Generate binary mask first (1=known, 0=unknown), shape (B, 1, H, W).
            # We keep it as a separate variable so we can use it to weight the loss.
            mask_binary = diffusion.create_random_mask(images.shape[0]).to(device)

            # FIX 2: Normalize mask to [-1, 1] for the model input channel.
            # The redundant `if mask.shape[1] > 1` branch has been removed —
            # create_random_mask always returns shape (B, 1, H, W), so the slice
            # was dead code and never executed.
            mask_norm = (mask_binary * 2.0) - 1.0  # shape: (B, 1, H, W)

            model_input = torch.cat([x_t, mask_norm], dim=1)
            predicted_noise = model(model_input, t)

            # Compute loss ONLY on the unknown region, normalized by the actual
            # number of unknown pixels. nn.MSELoss() would divide by B*C*H*W
            # regardless of mask size, causing gradient scale to vary wildly
            # between batches with large vs. small masks.
            unknown_region = 1.0 - mask_binary  # 1 where pixels are missing
            num_unknown = unknown_region.sum().clamp(min=1.0)
            sq_err = ((noise - predicted_noise) ** 2) * unknown_region
            loss = sq_err.sum() / num_unknown

            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"Batch {i}: loss is NaN or Inf ({loss}). Skipping optimizer step.")
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if torch.isnan(total_norm) or torch.isinf(total_norm):
                logging.warning(f"Batch {i}: Grad norm is {total_norm}. Skipping step.")
                continue

            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(MSE=f"{loss.item():.4f}", GradNorm=f"{total_norm:.2f}")

        avg_loss = epoch_loss / len(dataloader)
        logging.info(f"Epoch {epoch+1} Average Loss: {avg_loss:.6f}")

        if (epoch + 1) % 5 == 0:
            save_path = os.path.join("model.pt")
            checkpoint = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save(checkpoint, save_path)
            logging.info(f"Saved clean checkpoint to {save_path}")


def get_data(args):
    transforms_list = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Images to [-1, 1]
    ])
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {args.dataset_path}")
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
    batch_size = 48
    image_size = 64
    dataset_path = "/work/csse463/202620/06/diffusion-restoration/data/celeba_raw/img_align_celeba"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    lr = 3e-4
    epochs = 500


if __name__ == "__main__":
    args = Args()
    train(args)
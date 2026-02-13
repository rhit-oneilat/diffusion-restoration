import torch
import os
from torchvision import transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from src.unet import UNet
from src.diffusionInpainting import InpaintingDiffusion
from src.CelebADataset import CelebADataset
from collections import OrderedDict
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")


def test_inpainting(model_path, data_path, num_samples=8, mask_type='center'):
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    img_size = 64

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = CelebADataset(data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=True)

    model = UNet(c_in=4, device=device).to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Weights not found at {model_path}")

    state_dict = torch.load(model_path, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    diffusion = InpaintingDiffusion(img_size=img_size, device=device)

    real_images, _ = next(iter(dataloader))
    real_images = real_images.to(device)

    # binary_mask: 1=known, 0=unknown — shape (B, 1, H, W)
    binary_mask = diffusion.create_random_mask(num_samples, mask_type=mask_type).to(device)

    # FIX 1: Normalize mask to [-1, 1] for the model input, matching how it was
    # trained. The old code passed mask_norm into sample_with_inpainting but the
    # function was renamed to known_region_mask_norm to make the contract explicit.
    mask_norm = (binary_mask * 2.0) - 1.0

    # FIX 2: Build the masked display image correctly.
    # real_images is in [-1, 1]. Multiplying by binary_mask zeros out unknown pixels,
    # but 0 in [-1, 1] space renders as mid-grey, not black. Instead, fill unknown
    # regions with -1 (which denorms to 0.0 = black) so the visualization is correct.
    masked_images = real_images * binary_mask + (-1.0) * (1.0 - binary_mask)

    with torch.no_grad():
        restored_images_uint8 = diffusion.sample_with_inpainting(
            model,
            n=num_samples,
            known_region_mask_norm=mask_norm,   # FIX 3: updated parameter name
            known_region_data=real_images,
            repaint_jumps=10
        )

    # Convert uint8 output [0, 255] back to float [-1, 1] for consistent denorm
    restored_images = (restored_images_uint8.float() / 255.0) * 2.0 - 1.0

    def denorm(x):
        return (x.clamp(-1, 1) + 1) / 2

    # Stack rows: original | masked input | restored
    comparison = torch.cat([
        denorm(real_images),
        denorm(masked_images),
        denorm(restored_images)
    ], dim=0)

    grid = make_grid(comparison, nrow=num_samples)
    save_image(grid, "inpainting_results.png")
    logging.info("Results saved to inpainting_results.png")


if __name__ == "__main__":
    MODEL_PATH = "model.pt"
    DATA_PATH = "/work/csse463/202620/06/diffusion-restoration/data/celeba_raw/img_align_celeba"
    test_inpainting(MODEL_PATH, DATA_PATH, num_samples=4, mask_type='center')
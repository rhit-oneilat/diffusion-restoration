import torch
import torch.nn as nn
import torch.optim as optim
from src.unet import UNet
from src.diffusionInpainting import InpaintingDiffusion


def quick_train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 4
    img_size = 64
    steps = 30
    lr = 1e-3

    model = UNet(c_in=4, device=device).to(device)
    diffusion = InpaintingDiffusion(img_size=img_size, device=device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    # Synthetic fixed image batch in [-1, 1]
    x_real = (torch.rand(batch_size, 3, img_size, img_size) * 2.0 - 1.0).to(device)

    model.train()
    for step in range(steps):
        t = diffusion.sample_timesteps(batch_size).to(device)
        x_t, noise = diffusion.noise_images(x_real, t)
        mask = diffusion.create_random_mask(batch_size).to(device)
        model_input = torch.cat([x_t, mask], dim=1)

        predicted_noise = model(model_input, t)
        loss = mse(noise, predicted_noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 5 == 0:
            print(f"Step {step}: loss={loss.item():.6f}")

    print(f"Final loss after {steps} steps: {loss.item():.6f}")


if __name__ == '__main__':
    quick_train()

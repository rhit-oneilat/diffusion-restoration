import torch
import logging
from src.diffusion import DiffusionModule
import random

class InpaintingDiffusion(DiffusionModule):
    """Extends DiffusionModule with repainting for inpainting tasks"""
    
    def create_random_mask(self, batch_size, mask_type='random', mask_ratio=0.5):
        H, W = self.img_size, self.img_size
        masks = torch.ones((batch_size, 1, H, W))
        
        for i in range(batch_size):
            if mask_type == 'center':
                # Center square mask
                size = int(H * mask_ratio)
                start = (H - size) // 2
                masks[i, :, start:start+size, start:start+size] = 0
                
            elif mask_type == 'random':
                # Random rectangular region
                h = random.randint(int(H * 0.2), int(H * 0.6))
                w = random.randint(int(W * 0.2), int(W * 0.6))
                y = random.randint(0, H - h)
                x = random.randint(0, W - w)
                masks[i, :, y:y+h, x:x+w] = 0
                
            elif mask_type == 'half':
                # Random half (top, bottom, left, or right)
                side = random.choice(['top', 'bottom', 'left', 'right'])
                if side == 'top':
                    masks[i, :, :H//2, :] = 0
                elif side == 'bottom':
                    masks[i, :, H//2:, :] = 0
                elif side == 'left':
                    masks[i, :, :, :W//2] = 0
                else:  # right
                    masks[i, :, :, W//2:] = 0
                    
            elif mask_type == 'multiple_boxes':
                # Multiple random boxes
                num_boxes = random.randint(2, 5)
                for _ in range(num_boxes):
                    h = random.randint(int(H * 0.1), int(H * 0.3))
                    w = random.randint(int(W * 0.1), int(W * 0.3))
                    y = random.randint(0, H - h)
                    x = random.randint(0, W - w)
                    masks[i, :, y:y+h, x:x+w] = 0
                    
            elif mask_type == 'free_form':
                # Free-form brush strokes (like natural scratches/damage)
                num_strokes = random.randint(5, 15)
                for _ in range(num_strokes):
                    # Random start point
                    y, x = random.randint(0, H-1), random.randint(0, W-1)
                    # Random stroke length and thickness
                    length = random.randint(10, 30)
                    thickness = random.randint(2, 8)
                    
                    for step in range(length):
                        # Random walk
                        dy = random.randint(-2, 2)
                        dx = random.randint(-2, 2)
                        y = max(0, min(H-1, y + dy))
                        x = max(0, min(W-1, x + dx))
                        
                        # Draw thick point
                        y1 = max(0, y - thickness//2)
                        y2 = min(H, y + thickness//2)
                        x1 = max(0, x - thickness//2)
                        x2 = min(W, x + thickness//2)
                        masks[i, :, y1:y2, x1:x2] = 0
        
        return masks
    
    def sample_with_inpainting(self, model, n, known_region_mask, known_region_data, repaint_jumps=10):
        logging.info(f"Sampling {n} images with inpainting (repaint_jumps={repaint_jumps})...")
        model.eval()
        
        known_region_mask = known_region_mask.to(self.device)
        known_region_data = known_region_data.to(self.device)
        
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n) * i).long().to(self.device)
                
                # Repainting loop
                for r in range(repaint_jumps if i > 1 else 1):
                    # Concatenate known-region mask as extra channel for conditioning
                    model_input = torch.cat([x, known_region_mask], dim=1)
                    predicted_noise = model(model_input, t)
                    
                    alpha = self.alpha[t][:, None, None, None]
                    alpha_hat = self.alpha_hat[t][:, None, None, None]
                    beta = self.beta[t][:, None, None, None]
                    
                    # Denoising step
                    if i > 1:
                        noise = torch.randn_like(x)
                    else:
                        noise = torch.zeros_like(x)
                    
                    x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                    
                    # Repainting: replace known region
                    if r < repaint_jumps - 1 and i > 1:
                        t_prev = t - 1
                        known_noised, _ = self.noise_images(known_region_data, t_prev)
                        x = known_region_mask * known_noised + (1 - known_region_mask) * x
   
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

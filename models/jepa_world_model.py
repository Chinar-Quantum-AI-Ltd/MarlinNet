import numpy as np
import torch
import torch.nn as nn
import timm

VISUAL_DIM = 384
VESSEL_DIM = 3          # [speed, sin(heading), cos(heading)]
LATENT_DIM = VISUAL_DIM + VESSEL_DIM   # 387

class JEPAWorldModel(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        try:
            self.encoder = timm.create_model('vit_small_patch8_224.dino', pretrained=True, num_classes=0)
        except:
            # Lightweight fallback when DINO weights are unavailable.
            # Linear projects conv features to VISUAL_DIM so the latent shape
            # contract (LATENT_DIM = VISUAL_DIM + VESSEL_DIM) always holds.
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, VISUAL_DIM),
            )
        self.encoder.eval().to(device)
        for p in self.encoder.parameters():
            p.requires_grad = False

    def get_latent(self, obs_np, speed, heading_deg):
        x = torch.from_numpy(obs_np).unsqueeze(0).float().to(self.device)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        with torch.no_grad():
            visual = self.encoder(x).squeeze(0)   # (384,)

        # encode heading as sin/cos to avoid 0/360 discontinuity
        rad = np.radians(heading_deg)
        vessel_state = torch.tensor(
            [speed, np.sin(rad), np.cos(rad)],
            dtype=torch.float32, device=self.device
        )
        return torch.cat([visual, vessel_state], dim=0)   # (387,)

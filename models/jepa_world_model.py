import torch
import torch.nn as nn
import timm  # More reliable than torch.hub for DINO

class JEPAWorldModel(nn.Module):
    def __init__(self, device='mps'):
        super().__init__()
        self.device = device
        try:
            # Try timm DINO (more stable)
            self.encoder = timm.create_model('vit_small_patch8_224.dino', pretrained=True, num_classes=0)
        except:
            # Fallback to simple CNN if timm fails
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )
        self.encoder.eval().to(device)
        for p in self.encoder.parameters():
            p.requires_grad = False

    def get_latent(self, obs_np):
        x = torch.from_numpy(obs_np).unsqueeze(0).float().to(self.device)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        with torch.no_grad():
            feats = self.encoder(x)
            return feats.squeeze(0)  # 1D latent

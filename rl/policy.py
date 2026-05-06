import torch
import torch.nn as nn

class LatentPolicy(nn.Module):
    def __init__(self, latent_dim=384, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, latent):
        return torch.tanh(self.net(latent))

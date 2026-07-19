import torch
import torch.nn as nn
from models.jepa_world_model import LATENT_DIM

ACTION_DIM = 2
HIDDEN_DIM = 256


class GRUTransitionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(
            input_size=LATENT_DIM + ACTION_DIM,   # 389
            hidden_size=HIDDEN_DIM,
            num_layers=1,
            batch_first=True,
        )
        self.output_proj = nn.Linear(HIDDEN_DIM, LATENT_DIM)

    def forward(self, z_t, action_t, h_t=None):
        """
        z_t      : (batch, LATENT_DIM)   current latent
        action_t : (batch, ACTION_DIM)   left/right thruster
        h_t      : (1, batch, HIDDEN_DIM) GRU hidden state, None = zeros

        Returns:
          z_t1   : (batch, LATENT_DIM)   predicted next latent
          h_t1   : (1, batch, HIDDEN_DIM) updated hidden state
        """
        x = torch.cat([z_t, action_t], dim=-1).unsqueeze(1)  # (batch, 1, 389)
        out, h_t1 = self.gru(x, h_t)                         # out: (batch, 1, 256)
        z_t1 = self.output_proj(out.squeeze(1))               # (batch, LATENT_DIM)
        return z_t1, h_t1

    def init_hidden(self, batch_size=1, device='cuda'):
        return torch.zeros(1, batch_size, HIDDEN_DIM, device=device)

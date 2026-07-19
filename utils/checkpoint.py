import torch

def save_checkpoint(model, path="checkpoints/marlinnet_latest.pth"):
    torch.save(model.state_dict(), path)
    print(f"💾 Checkpoint saved: {path}")

def load_checkpoint(model, path):
    model.load_state_dict(torch.load(path, weights_only=True))
    print(f"📂 Checkpoint loaded: {path}")

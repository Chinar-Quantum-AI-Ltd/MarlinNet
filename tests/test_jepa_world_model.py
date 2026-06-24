import numpy as np
import pytest
import torch
import torch.nn as nn
from models.jepa_world_model import JEPAWorldModel, LATENT_DIM, VISUAL_DIM, VESSEL_DIM

DEVICE = "cpu"


class FakeEncoder(nn.Module):
    """Lightweight stand-in for the frozen DINO ViT — returns correct visual dim."""
    def forward(self, x):
        return torch.zeros(x.shape[0], VISUAL_DIM)


@pytest.fixture
def model(monkeypatch):
    import timm
    monkeypatch.setattr(timm, "create_model", lambda *a, **kw: FakeEncoder())
    return JEPAWorldModel(device=DEVICE)


@pytest.fixture
def blank_obs():
    return np.zeros((3, 224, 224), dtype=np.float32)


class TestGetLatentOutputShape:
    def test_latent_dim_is_correct(self, model, blank_obs):
        z = model.get_latent(blank_obs, speed=5.0, heading_deg=0.0)
        assert z.shape == (LATENT_DIM,), f"Expected ({LATENT_DIM},), got {z.shape}"

    def test_visual_plus_vessel_dims(self):
        assert LATENT_DIM == VISUAL_DIM + VESSEL_DIM
        assert VISUAL_DIM == 384
        assert VESSEL_DIM == 3

    def test_output_is_on_correct_device(self, model, blank_obs):
        z = model.get_latent(blank_obs, speed=5.0, heading_deg=0.0)
        assert z.device.type == DEVICE

    def test_grayscale_obs_expands_to_rgb(self, model):
        grey_obs = np.zeros((1, 224, 224), dtype=np.float32)
        z = model.get_latent(grey_obs, speed=5.0, heading_deg=0.0)
        assert z.shape == (LATENT_DIM,)


class TestVesselStateEmbedding:
    def test_speed_embedded_in_last_3_dims(self, model, blank_obs):
        speed = 7.5
        z = model.get_latent(blank_obs, speed=speed, heading_deg=0.0)
        assert z[-3].item() == pytest.approx(speed, rel=1e-5)

    def test_heading_sin_cos_embedded(self, model, blank_obs):
        heading = 90.0
        z = model.get_latent(blank_obs, speed=0.0, heading_deg=heading)
        assert z[-2].item() == pytest.approx(np.sin(np.radians(heading)), abs=1e-5)
        assert z[-1].item() == pytest.approx(np.cos(np.radians(heading)), abs=1e-5)

    @pytest.mark.parametrize("heading,exp_sin,exp_cos", [
        (0.0,   0.0,  1.0),
        (90.0,  1.0,  0.0),
        (180.0, 0.0, -1.0),
        (270.0,-1.0,  0.0),
        (360.0, 0.0,  1.0),   # 360° wraps to same as 0°
    ])
    def test_heading_encoding_cardinal_directions(self, model, blank_obs, heading, exp_sin, exp_cos):
        z = model.get_latent(blank_obs, speed=0.0, heading_deg=heading)
        assert z[-2].item() == pytest.approx(exp_sin, abs=1e-5)
        assert z[-1].item() == pytest.approx(exp_cos, abs=1e-5)

    def test_different_headings_produce_different_latents(self, model, blank_obs):
        z0 = model.get_latent(blank_obs, speed=5.0, heading_deg=0.0)
        z90 = model.get_latent(blank_obs, speed=5.0, heading_deg=90.0)
        assert not torch.allclose(z0, z90)

    def test_different_speeds_produce_different_latents(self, model, blank_obs):
        z_slow = model.get_latent(blank_obs, speed=0.0, heading_deg=0.0)
        z_fast = model.get_latent(blank_obs, speed=20.0, heading_deg=0.0)
        assert not torch.allclose(z_slow, z_fast)

    def test_encoder_is_frozen(self, model):
        for p in model.encoder.parameters():
            assert not p.requires_grad, "Encoder weights must be frozen"

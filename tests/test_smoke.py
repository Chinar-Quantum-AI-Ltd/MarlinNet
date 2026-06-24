"""
Smoke tests: end-to-end pipeline for a handful of steps.
Heavy external models (DINO ViT, YOLO) are replaced with lightweight stand-ins
so these tests run fast without network access or GPU.
"""
import os
import numpy as np
import pytest
import torch
import torch.nn as nn

from env.maritime_env import MaritimeEnv
from models.transition_model import GRUTransitionModel, LATENT_DIM, ACTION_DIM
from models.jepa_world_model import VISUAL_DIM

DEVICE = "cpu"


class FakeEncoder(nn.Module):
    """Returns a correctly-shaped visual embedding without loading DINO weights."""
    def forward(self, x):
        return torch.zeros(x.shape[0], VISUAL_DIM)


@pytest.fixture(scope="module")
def world_model(monkeypatch_module):
    import timm
    monkeypatch_module.setattr(timm, "create_model", lambda *a, **kw: FakeEncoder())
    from models.jepa_world_model import JEPAWorldModel
    return JEPAWorldModel(device=DEVICE)


@pytest.fixture(scope="module")
def transition_model():
    m = GRUTransitionModel().to(DEVICE)
    m.eval()
    return m


@pytest.fixture(scope="module")
def env():
    e = MaritimeEnv()
    e.reset()
    return e


# module-scoped monkeypatch shim (pytest's built-in monkeypatch is function-scoped)
@pytest.fixture(scope="module")
@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


class TestFullPipelineSmoke:
    def test_five_step_loop(self, env, world_model, transition_model):
        obs, _ = env.reset()
        h_t = transition_model.init_hidden(batch_size=1, device=DEVICE)

        for step in range(5):
            speed, heading = env.get_vessel_state()
            z_t = world_model.get_latent(obs, speed, heading)

            assert z_t.shape == (LATENT_DIM,), f"Step {step}: bad latent shape"
            assert torch.isfinite(z_t).all(), f"Step {step}: NaN/Inf in z_t"

            action = env.action_space.sample()
            obs, reward, done, _, _ = env.step(action)

            action_t = torch.tensor(action, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                z_t1_pred, h_t = transition_model(z_t.unsqueeze(0), action_t, h_t)

            assert z_t1_pred.shape == (1, LATENT_DIM), f"Step {step}: bad prediction shape"
            assert torch.isfinite(z_t1_pred).all(), f"Step {step}: NaN/Inf in z_t1_pred"
            assert h_t.shape == (1, 1, 256), f"Step {step}: bad hidden state shape"

            if done:
                break

    def test_z_t1_norm_is_nonzero(self, env, world_model, transition_model):
        obs, _ = env.reset()
        speed, heading = env.get_vessel_state()
        z_t = world_model.get_latent(obs, speed, heading)
        action_t = torch.zeros(1, ACTION_DIM, device=DEVICE)
        h0 = transition_model.init_hidden(1, DEVICE)
        with torch.no_grad():
            z_t1, _ = transition_model(z_t.unsqueeze(0), action_t, h0)
        assert z_t1.norm().item() > 0.0, "Predicted latent should not be all-zeros"

    def test_hidden_state_is_deterministic_in_eval(self, env, world_model, transition_model):
        obs, _ = env.reset()
        speed, heading = env.get_vessel_state()
        z_t = world_model.get_latent(obs, speed, heading).unsqueeze(0)
        action_t = torch.zeros(1, ACTION_DIM, device=DEVICE)
        h0 = transition_model.init_hidden(1, DEVICE)

        with torch.no_grad():
            out_a, _ = transition_model(z_t, action_t, h0)
            out_b, _ = transition_model(z_t, action_t, h0)

        assert torch.allclose(out_a, out_b), "Model must be deterministic in eval mode"

    def test_multi_step_hidden_context(self, env, world_model, transition_model):
        """Prediction at step N should differ based on whether step N-1 context is carried."""
        obs, _ = env.reset()
        speed, heading = env.get_vessel_state()
        z1 = world_model.get_latent(obs, speed, heading).unsqueeze(0)

        obs, *_ = env.step(env.action_space.sample())
        speed, heading = env.get_vessel_state()
        z2 = world_model.get_latent(obs, speed, heading).unsqueeze(0)

        action_t = torch.zeros(1, ACTION_DIM, device=DEVICE)
        h0 = transition_model.init_hidden(1, DEVICE)

        with torch.no_grad():
            _, h1 = transition_model(z1, action_t, h0)
            out_carried, _ = transition_model(z2, action_t, h1)  # context from step 1
            out_fresh, _ = transition_model(z2, action_t, h0)    # no prior context

        assert not torch.allclose(out_carried, out_fresh), \
            "Prediction should differ when prior hidden context is carried"


class TestCheckpointSmoke:
    def test_save_and_load_transition_model(self, transition_model, tmp_path):
        from utils.checkpoint import save_checkpoint, load_checkpoint

        ckpt_path = str(tmp_path / "transition_smoke.pth")
        save_checkpoint(transition_model, ckpt_path)
        assert os.path.exists(ckpt_path), "Checkpoint file not created"

        loaded = GRUTransitionModel().to(DEVICE)
        load_checkpoint(loaded, ckpt_path)

        for p_orig, p_loaded in zip(transition_model.parameters(), loaded.parameters()):
            assert torch.allclose(p_orig, p_loaded), "Loaded weights differ from saved"

    def test_checkpoint_is_nonzero_bytes(self, transition_model, tmp_path):
        from utils.checkpoint import save_checkpoint

        ckpt_path = str(tmp_path / "transition_size_check.pth")
        save_checkpoint(transition_model, ckpt_path)
        assert os.path.getsize(ckpt_path) > 0

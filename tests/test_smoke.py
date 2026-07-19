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


class TestRealEncoderSmoke:
    """Tests that run through the fallback conv encoder — no FakeEncoder stub."""

    @pytest.fixture
    def real_world_model(self, monkeypatch):
        """Force the fallback path by making timm.create_model raise, then
        instantiate JEPAWorldModel so the real conv encoder is used."""
        import timm
        monkeypatch.setattr(
            timm, "create_model",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("timm unavailable"))
        )
        from models.jepa_world_model import JEPAWorldModel
        return JEPAWorldModel(device=DEVICE)

    def test_fallback_encoder_output_dim(self, real_world_model):
        obs = np.random.rand(3, 224, 224).astype(np.float32)
        z = real_world_model.get_latent(obs, speed=5.0, heading_deg=45.0)
        assert z.shape == (LATENT_DIM,), (
            f"Fallback encoder produced wrong latent dim: {z.shape}. "
            "Check that the fallback conv projects to VISUAL_DIM."
        )

    def test_fallback_encoder_produces_finite_values(self, real_world_model):
        obs = np.random.rand(3, 224, 224).astype(np.float32)
        z = real_world_model.get_latent(obs, speed=5.0, heading_deg=45.0)
        assert torch.isfinite(z).all(), "Fallback encoder produced NaN/Inf in latent"

    def test_fallback_encoder_nonzero_visual_component(self, real_world_model):
        """Real conv features should not be all-zeros (unlike FakeEncoder)."""
        obs = np.random.rand(3, 224, 224).astype(np.float32)
        z = real_world_model.get_latent(obs, speed=0.0, heading_deg=0.0)
        visual = z[:-3]
        assert visual.norm().item() > 0.0, "Fallback encoder visual component is all-zeros"

    def test_fallback_full_pipeline_five_steps(self, real_world_model):
        """End-to-end pipeline using the real fallback encoder for 5 steps."""
        transition_model = GRUTransitionModel().to(DEVICE)
        transition_model.eval()
        env = MaritimeEnv()
        obs, _ = env.reset()
        h_t = transition_model.init_hidden(batch_size=1, device=DEVICE)

        for step in range(5):
            speed, heading = env.get_vessel_state()
            z_t = real_world_model.get_latent(obs, speed, heading)
            assert z_t.shape == (LATENT_DIM,), f"Step {step}: wrong latent shape"

            action = env.action_space.sample()
            obs, _, done, _, _ = env.step(action)

            action_t = torch.tensor(action, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                z_t1_pred, h_t = transition_model(z_t.unsqueeze(0), action_t, h_t)

            assert torch.isfinite(z_t1_pred).all(), f"Step {step}: NaN/Inf in prediction"
            if done:
                break


class TestGRUHiddenStateEpisodeContract:
    """Documents the intended h_t lifecycle: persist within episode, reset on done."""

    def test_hidden_state_persists_across_steps_within_episode(
        self, env, world_model, transition_model
    ):
        """h_t must be passed through each step — not re-initialised mid-episode."""
        obs, _ = env.reset()
        h_t = transition_model.init_hidden(1, DEVICE)
        action_t = torch.zeros(1, ACTION_DIM, device=DEVICE)

        h_after_step1 = None
        for i in range(3):
            speed, heading = env.get_vessel_state()
            z_t = world_model.get_latent(obs, speed, heading)
            obs, _, done, _, _ = env.step(env.action_space.sample())
            with torch.no_grad():
                _, h_t = transition_model(z_t.unsqueeze(0), action_t, h_t)
            if i == 0:
                h_after_step1 = h_t.clone()
            if done:
                break

        # h_t after 3 steps must differ from h_t after 1 step
        assert not torch.allclose(h_t, h_after_step1), \
            "h_t should accumulate context across steps — it should change each step"

    def test_reset_on_done_differs_from_carrying_terminal_state(
        self, env, world_model, transition_model
    ):
        """Carrying terminal h_t into a new episode must produce a different
        prediction than resetting with init_hidden — enforces the reset-on-done contract."""
        obs, _ = env.reset()
        h_t = transition_model.init_hidden(1, DEVICE)
        action_t = torch.zeros(1, ACTION_DIM, device=DEVICE)

        # Run a few steps to build up hidden state context
        for _ in range(5):
            speed, heading = env.get_vessel_state()
            z_t = world_model.get_latent(obs, speed, heading)
            obs, _, done, _, _ = env.step(env.action_space.sample())
            with torch.no_grad():
                _, h_t = transition_model(z_t.unsqueeze(0), action_t, h_t)
            if done:
                break

        # Simulate start of a new episode
        obs, _ = env.reset()
        speed, heading = env.get_vessel_state()
        z_new_ep = world_model.get_latent(obs, speed, heading).unsqueeze(0)

        h_fresh = transition_model.init_hidden(1, DEVICE)

        with torch.no_grad():
            out_carried, _ = transition_model(z_new_ep, action_t, h_t)     # wrong: leaked state
            out_reset, _ = transition_model(z_new_ep, action_t, h_fresh)   # correct: episode reset

        assert not torch.allclose(out_carried, out_reset), \
            "reset-on-done contract: init_hidden at episode boundary must change predictions"


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

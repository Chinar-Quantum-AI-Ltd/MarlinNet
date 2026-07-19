import pytest
import torch
from models.transition_model import GRUTransitionModel, LATENT_DIM, ACTION_DIM, HIDDEN_DIM

DEVICE = "cpu"


@pytest.fixture
def model():
    return GRUTransitionModel().to(DEVICE)


class TestGRUTransitionModelShapes:
    def test_output_shape_batch1(self, model):
        z = torch.randn(1, LATENT_DIM)
        a = torch.randn(1, ACTION_DIM)
        z_next, h_next = model(z, a)
        assert z_next.shape == (1, LATENT_DIM)
        assert h_next.shape == (1, 1, HIDDEN_DIM)

    def test_output_shape_batch4(self, model):
        B = 4
        z = torch.randn(B, LATENT_DIM)
        a = torch.randn(B, ACTION_DIM)
        z_next, h_next = model(z, a)
        assert z_next.shape == (B, LATENT_DIM)
        assert h_next.shape == (1, B, HIDDEN_DIM)

    def test_init_hidden_shape_batch1(self, model):
        h = model.init_hidden(batch_size=1, device=DEVICE)
        assert h.shape == (1, 1, HIDDEN_DIM)

    def test_init_hidden_shape_batch4(self, model):
        h = model.init_hidden(batch_size=4, device=DEVICE)
        assert h.shape == (1, 4, HIDDEN_DIM)

    def test_init_hidden_is_zeros(self, model):
        h = model.init_hidden(batch_size=2, device=DEVICE)
        assert torch.all(h == 0)

    def test_init_hidden_on_correct_device(self, model):
        h = model.init_hidden(batch_size=1, device=DEVICE)
        assert h.device.type == DEVICE


class TestGRUTransitionModelBehavior:
    def test_none_hidden_accepted(self, model):
        z = torch.randn(1, LATENT_DIM)
        a = torch.randn(1, ACTION_DIM)
        z_next, h_next = model(z, a, h_t=None)
        assert z_next.shape == (1, LATENT_DIM)

    def test_explicit_hidden_same_as_none(self, model):
        """Passing zeros explicitly should match h_t=None (GRU default is zeros)."""
        model.eval()
        z = torch.randn(1, LATENT_DIM)
        a = torch.randn(1, ACTION_DIM)
        h0 = model.init_hidden(batch_size=1, device=DEVICE)
        with torch.no_grad():
            z_explicit, _ = model(z, a, h0)
            z_none, _ = model(z, a, None)
        assert torch.allclose(z_explicit, z_none)

    def test_hidden_state_changes_after_step(self, model):
        z = torch.randn(1, LATENT_DIM)
        a = torch.randn(1, ACTION_DIM)
        h0 = model.init_hidden(batch_size=1, device=DEVICE)
        _, h1 = model(z, a, h0)
        assert not torch.allclose(h0, h1), "Hidden state should change after a forward pass"

    def test_carried_hidden_differs_from_fresh(self, model):
        """Two identical steps with a carried hidden state differ from two independent steps."""
        model.eval()
        z1 = torch.randn(1, LATENT_DIM)
        z2 = torch.randn(1, LATENT_DIM)
        a = torch.randn(1, ACTION_DIM)
        h0 = model.init_hidden(1, DEVICE)

        with torch.no_grad():
            _, h1 = model(z1, a, h0)
            z_carried, _ = model(z2, a, h1)   # uses context from step 1
            z_fresh, _ = model(z2, a, h0)     # fresh hidden, no step-1 context

        assert not torch.allclose(z_carried, z_fresh), \
            "Output should differ when hidden state carries prior context"

    def test_different_actions_give_different_outputs(self, model):
        model.eval()
        z = torch.randn(1, LATENT_DIM)
        a_full_ahead = torch.ones(1, ACTION_DIM)
        a_full_reverse = -torch.ones(1, ACTION_DIM)
        h = model.init_hidden(1, DEVICE)
        with torch.no_grad():
            z1, _ = model(z, a_full_ahead, h)
            z2, _ = model(z, a_full_reverse, h)
        assert not torch.allclose(z1, z2)

    def test_output_is_finite(self, model):
        z = torch.randn(1, LATENT_DIM)
        a = torch.randn(1, ACTION_DIM)
        z_next, h_next = model(z, a)
        assert torch.isfinite(z_next).all()
        assert torch.isfinite(h_next).all()

    def test_deterministic_in_eval_mode(self, model):
        model.eval()
        z = torch.randn(1, LATENT_DIM)
        a = torch.randn(1, ACTION_DIM)
        h = model.init_hidden(1, DEVICE)
        with torch.no_grad():
            out_a, _ = model(z, a, h)
            out_b, _ = model(z, a, h)
        assert torch.allclose(out_a, out_b)

    def test_backward_pass_does_not_error(self, model):
        z = torch.randn(1, LATENT_DIM, requires_grad=True)
        a = torch.randn(1, ACTION_DIM)
        z_next, _ = model(z, a)
        z_next.sum().backward()
        assert z.grad is not None
        assert torch.isfinite(z.grad).all()

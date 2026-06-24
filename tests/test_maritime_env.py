import numpy as np
import pytest
from env.maritime_env import MaritimeEnv


@pytest.fixture
def env():
    e = MaritimeEnv()
    e.reset()
    return e


class TestGetVesselState:
    def test_returns_two_values(self, env):
        result = env.get_vessel_state()
        assert len(result) == 2

    def test_initial_speed_is_5(self, env):
        speed, _ = env.get_vessel_state()
        assert speed == pytest.approx(5.0)

    def test_initial_heading_is_0(self, env):
        _, heading = env.get_vessel_state()
        assert heading == pytest.approx(0.0)

    def test_reflects_state_after_step(self, env):
        speed_before, heading_before = env.get_vessel_state()
        # Asymmetric action changes heading; symmetric action changes speed.
        env.step(np.array([1.0, -1.0], dtype=np.float32))
        speed_after, heading_after = env.get_vessel_state()
        assert speed_after != speed_before or heading_after != heading_before

    def test_heading_stays_in_0_360_range(self, env):
        for _ in range(200):
            env.step(env.action_space.sample())
            _, heading = env.get_vessel_state()
            assert 0.0 <= heading < 360.0

    def test_speed_is_clipped_high(self, env):
        env.step(np.array([1.0, 1.0], dtype=np.float32))
        speed, _ = env.get_vessel_state()
        assert speed <= 20.0

    def test_speed_is_clipped_low(self, env):
        env.step(np.array([-1.0, -1.0], dtype=np.float32))
        speed, _ = env.get_vessel_state()
        assert speed >= 0.0

    def test_reset_restores_defaults(self):
        e = MaritimeEnv()
        for _ in range(10):
            e.step(e.action_space.sample())
        e.reset()
        speed, heading = e.get_vessel_state()
        assert speed == pytest.approx(5.0)
        assert heading == pytest.approx(0.0)

    @pytest.mark.parametrize("action,expected_speed", [
        ([1.0, 1.0], 10.0),    # (1+1)/2 * 10
        ([0.0, 0.0], 0.0),     # (0+0)/2 * 10
        ([0.5, 0.5], 5.0),     # (0.5+0.5)/2 * 10
    ])
    def test_speed_formula(self, action, expected_speed):
        e = MaritimeEnv()
        e.reset()
        e.step(np.array(action, dtype=np.float32))
        speed, _ = e.get_vessel_state()
        assert speed == pytest.approx(expected_speed, abs=1e-5)


class TestMaritimeEnvStep:
    def test_step_returns_five_values(self, env):
        result = env.step(env.action_space.sample())
        assert len(result) == 5

    def test_obs_shape(self, env):
        obs, *_ = env.step(env.action_space.sample())
        assert obs.shape == (3, 224, 224)

    def test_obs_dtype(self, env):
        obs, *_ = env.step(env.action_space.sample())
        assert obs.dtype == np.float32

    def test_truncated_is_always_false(self, env):
        _, _, _, truncated, _ = env.step(env.action_space.sample())
        assert truncated is False

    def test_done_when_goal_reached(self):
        e = MaritimeEnv()
        e.reset()
        e.pos = np.array([180.0, 180.0])   # place vessel at goal
        _, _, done, _, _ = e.step(np.array([0.0, 0.0], dtype=np.float32))
        assert done

    def test_done_when_out_of_bounds(self):
        e = MaritimeEnv()
        e.reset()
        e.pos = np.array([-1.0, 100.0])    # already out of bounds
        _, _, done, _, _ = e.step(np.array([0.0, 0.0], dtype=np.float32))
        assert done

    def test_reset_returns_obs_and_info(self):
        e = MaritimeEnv()
        obs, info = e.reset()
        assert obs.shape == (3, 224, 224)
        assert isinstance(info, dict)

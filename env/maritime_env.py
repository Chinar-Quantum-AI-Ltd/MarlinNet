import numpy as np
import gymnasium as gym
from gymnasium import spaces

class MaritimeEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(0, 255, (3, 224, 224), np.uint8)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.reset()

    def reset(self, seed=None):
        self.pos = np.array([20.0, 20.0])
        self.heading = 0.0
        self.speed = 5.0
        self.goal = np.array([180.0, 180.0])
        self.other_vessels = [np.random.rand(2) * 200 for _ in range(3)]
        return self._get_obs(), {}

    def _get_obs(self):
        return np.zeros((3, 224, 224), dtype=np.float32)

    def step(self, action):
        left, right = action
        self.speed = np.clip((left + right) / 2.0 * 10.0, 0, 20)
        self.heading = (self.heading + (right - left) * 5.0) % 360
        rad = np.radians(self.heading)
        self.pos += self.speed * np.array([np.cos(rad), np.sin(rad)]) * 0.08

        dist = np.linalg.norm(self.pos - self.goal)
        reward = -0.1 + (dist < 15) * 50
        done = dist < 10 or np.any(self.pos < 0) or np.any(self.pos > 200)
        return self._get_obs(), reward, done, False, {}

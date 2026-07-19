import os
import yaml
import torch
import numpy as np
from models.jepa_world_model import JEPAWorldModel
from models.transition_model import GRUTransitionModel
from env.maritime_env import MaritimeEnv
from planning.collision_avoidance import velocity_obstacle_avoidance
from rl.policy import LatentPolicy
from perception.waste_detector import detect_waste
from utils.checkpoint import save_checkpoint

with open("config.yaml") as f:
    config = yaml.safe_load(f)

device = config["device"]

env = MaritimeEnv()
world_model = JEPAWorldModel(device=device)
transition_model = GRUTransitionModel().to(device)
policy = LatentPolicy().to(device)

obs, _ = env.reset()
total_reward = 0.0
h_t = transition_model.init_hidden(batch_size=1, device=device)

print("🚢 MarlinNet - JEPA Maritime World Model Running...\n")

for step in range(config.get("max_steps", 500)):
    speed, heading = env.get_vessel_state()
    z_t = world_model.get_latent(obs, speed, heading)

    # RL policy acts on current latent
    action_rl = policy(z_t.unsqueeze(0)).detach().cpu().numpy()[0]

    # Classical collision avoidance overrides RL action for safety
    rad = np.radians(heading)
    own_vel = np.array([speed * np.cos(rad), speed * np.sin(rad)])
    action = velocity_obstacle_avoidance(env.pos, own_vel, env.other_vessels)

    obs, reward, done, _, info = env.step(action)
    total_reward += reward

    # GRU: predict next latent from current latent + action taken
    action_t = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        z_t1_pred, h_t = transition_model(z_t.unsqueeze(0), action_t, h_t)

    wastes = detect_waste(obs)

    if step % 30 == 0 or done:
        print(f"Step {step:3d} | Reward: {total_reward:6.1f} | "
              f"Wastes: {len(wastes)} | Latent: {z_t.shape[0]} | "
              f"z_t+1 norm: {z_t1_pred.norm().item():.2f}")

    if done:
        break

os.makedirs("checkpoints", exist_ok=True)
save_checkpoint(world_model, "checkpoints/marlinnet_final.pth")
save_checkpoint(transition_model, "checkpoints/transition_model_final.pth")
print(f"\nEpisode Finished! Total Reward: {total_reward:.1f}")

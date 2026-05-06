import yaml
import numpy as np
from models.jepa_world_model import JEPAWorldModel
from env.maritime_env import MaritimeEnv
from planning.collision_avoidance import velocity_obstacle_avoidance
from rl.policy import LatentPolicy
from perception.waste_detector import detect_waste
from utils.checkpoint import save_checkpoint

with open("config.yaml") as f:
    config = yaml.safe_load(f)

env = MaritimeEnv()
world_model = JEPAWorldModel(device=config["device"])
policy = LatentPolicy().to(config["device"])

obs, _ = env.reset()
total_reward = 0.0

print("🚢 MarlinNet - JEPA Maritime World Model Running...\n")

for step in range(config.get("max_steps", 500)):
    latent = world_model.get_latent(obs)
    
    # RL Policy + Classical Override
    action_rl = policy(latent.unsqueeze(0)).detach().cpu().numpy()[0]
    action = velocity_obstacle_avoidance(env.pos, env.other_vessels)
    
    obs, reward, done, _, info = env.step(action)
    total_reward += reward
    wastes = detect_waste(obs)
    
    if step % 30 == 0 or done:
        print(f"Step {step:3d} | Reward: {total_reward:6.1f} | "
              f"Wastes: {len(wastes)} | Latent: {latent.shape[0]}")
    
    if done:
        break

save_checkpoint(world_model, "checkpoints/marlinnet_final.pth")
print(f"\nEpisode Finished! Total Reward: {total_reward:.1f}")

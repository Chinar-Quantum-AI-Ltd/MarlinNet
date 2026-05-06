# MarlinNet 🛳️

**Joint Embedding Predictive Architecture (JEPA) World Model for Autonomous Maritime Navigation**

An end-to-end AI framework for autonomous vessels that combines:
- **Predictive World Modeling** using JEPA-style embeddings
- **Real-time Marine Waste Detection & Classification**
- **Collision Avoidance** (Velocity Obstacles + COLREGs compliance)
- **Reinforcement Learning** policy operating in latent space
- **Multimodal Sensor Fusion** ready architecture
---

## ✨ Key Features

- **JEPA-style World Model**: Learns rich, predictive latent representations from visual observations (DINO ViT backbone — ready for official Meta V-JEPA)
- **Waste Perception**: YOLOv8-based detection and classification of marine debris (plastic, nets, metal, etc.)
- **Safe Navigation**: Classical Velocity Obstacle avoidance with COLREGs-inspired maneuvers
- **Latent-space RL**: Policy network trained on JEPA embeddings (Stable-Baselines3 ready)
- **Apple Silicon Optimized**: Full MPS (Metal Performance Shaders) support
- **Modular & Extensible**: Clean separation of concerns for easy research and deployment
- **Checkpointing**: Save/load world model and policy weights
---

##  Core Components
### JEPA World Model (models/jepa_world_model.py)
- Uses DINO ViT-S/8 as strong proxy for Meta V-JEPA
- Produces compact latent embeddings (384-dim)
- Frozen backbone for efficiency

### Maritime Environment (env/maritime_env.py)
- 2D kinematic vessel model (Nomoto-like)
- Multiple dynamic obstacles
- Goal-directed navigation with reward shaping

### Waste Detection (perception/waste_detector.py)
- Real-time YOLOv8 inference
- Supports custom marine debris fine-tuned models

### Collision Avoidance (planning/collision_avoidance.py)
- Velocity Obstacle (VO) method
- COLREGs-inspired rule-based overrides (starboard turn, etc.)

### Latent Policy (rl/policy.py)
- MLP operating directly on JEPA embeddings
- Ready for PPO / SAC training with Stable-Baselines3

## Configuration
- device: mps
- yolo_model: yolov8s.pt
- max_steps: 500
- learning_rate: 1e-4

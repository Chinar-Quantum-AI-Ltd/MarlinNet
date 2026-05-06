import numpy as np

def velocity_obstacle_avoidance(own_pos, own_vel, others, safety_dist=15):
    """Simple VO + COLREGs-inspired avoidance"""
    for o_pos in others:
        rel = o_pos - own_pos
        dist = np.linalg.norm(rel)
        if dist < safety_dist:
            return np.array([0.6, -0.2])  # Hard starboard turn
    return np.array([0.0, 0.3])

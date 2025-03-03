import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from scipy.ndimage import gaussian_filter1d

from environment.env import CarFollowingEnv
from agent.softq import SoftQ

class Evaluator():
    def __init__(
            self,
            config: OmegaConf,
            environment: CarFollowingEnv,
            agent: SoftQ,
    ) -> SoftQ:
        self.config = config
        self.env = environment
        self.agent = agent

    @classmethod
    def sample_trajectory(
        cls,
        steps: int=500,
        min_v: int=10,
        max_v: int=30,
        smoothness: int=50,
    ) -> np.ndarray:

        changes = np.random.normal(0, 0.1, steps)
        walk = np.cumsum(changes)
        walk = (walk - walk.min()) / (walk.max() - walk.min())
        walk = walk * (max_v - min_v) + min_v
        smooth_walk = gaussian_filter1d(walk, sigma=smoothness)
        smooth_walk = np.clip(smooth_walk, min_v, max_v)
        return smooth_walk

    def evaluate(
        self,
        leader_trajectory: np.ndarray | None = None,
        v_ego_init: int = 20,
        d_ego_init: int = 30,
     ) -> None:

        if not leader_trajectory:
            leader_trajectory = self.sample_trajectory()

        follower_trajectory = []
        v_rel_init = leader_trajectory[0]-v_ego_init
        ego_vehicle_state = np.array([
            v_ego_init,
            d_ego_init,
            v_rel_init,
        ])
        for step in range(len(leader_trajectory) - 1):
            self.env.reset()
            self.env.state = ego_vehicle_state
            action = self.agent.choose_action(ego_vehicle_state)
            next_state, reward, terminated, truncated, _  = self.env.step(
                action=action,
                lead_speed=leader_trajectory[step+1]
            )
            follower_trajectory.append(next_state)
            ego_vehicle_state = next_state

        dummy_variable = follower_trajectory[-1]
        follower_trajectory.append(dummy_variable)

        follower_velocity = np.array([item[0] for item in follower_trajectory])
        distance_gap = np.array([item[1] for item in follower_trajectory])
        relative_speed = np.array([item[2] for item in follower_trajectory])
        time_steps = range(len(leader_trajectory))

        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, follower_velocity, label='follower velocity', marker='o', markersize=5)
        plt.plot(time_steps, leader_trajectory, label='leader velocity', marker='o', markersize=5)
        plt.xlabel('Time Steps in 0.1s')
        plt.ylabel('Velocity in m/s')
        plt.title('Velocity over time')
        plt.legend()
        plt.grid(True)

        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, distance_gap, label='distance gap', marker='o', markersize=5)
        plt.xlabel('Time steps in 0.1s')
        plt.ylabel('Distance gap in m')
        plt.title('Distance gap to lead vehicle')
        plt.legend()
        plt.grid(True)

        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, relative_speed, label='distance gap/velocity', marker='o', markersize=5)
        plt.xlabel('Time steps in 0.1s')
        plt.ylabel('relative velocity in m/s')
        plt.title('Relative velocity (v_lead - v_ego)')
        plt.legend()
        plt.grid(True)

        # Show the plot
        plt.show()
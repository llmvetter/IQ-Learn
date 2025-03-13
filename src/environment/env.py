from typing import Optional, Any

import gym
import numpy as np
import gymnasium as gym

from src.environment.simulator import Simulator

class CarFollowingEnv(gym.Env):
    def __init__(
            self,
            dataset_path: str,
            max_speed: int = 30,
            max_distance: int = 100,
            max_rel_speed: int = 75,
            actions: list[float]= [-0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2],
            delta_t: float = 0.1,
    ) -> None:
        # Environment parameters
        super().__init__()
        self.max_speed = max_speed
        self.max_distance = max_distance
        self.max_rel_speed = max_rel_speed
        self.delta_t = delta_t
        self.simulator = Simulator(path=dataset_path)

        self.action_space = gym.spaces.Discrete(len(actions))
        self.action_mapping = {i: actions[i] for i in range(len(actions))}

        # state: [ego speed, distance to lead vehicle]
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, -self.max_rel_speed]),
            high=np.array([
                self.max_speed,
                self.max_distance,
                self.max_rel_speed,
            ]),
            dtype=np.float32
        )
        self.state = None

    def reset(
            self, seed: Optional[int] = None,
            options: Optional[dict] = None,
    ) -> tuple[np.array, dict[str, Any]]:
        """Reset the environment to an initial state."""
        super().reset(seed=seed)

        self.state = np.array([
            np.random.uniform(5, self.max_speed-5),  # ego speed
            np.random.uniform(5, self.max_distance/2),  # distance to lead vehicle
            np.random.uniform(-self.max_rel_speed/2, self.max_rel_speed)
        ], dtype=np.float32)
        return self._get_obs(), self._get_info()

    def step(
            self,
            action: int,
            lead_speed: Optional[float] = None,
    ) -> None:
        """Take an action and return the next state, reward, done flag, and additional info."""
        ### RELATIVE SPEED = V_LEAD - V_FOLLOW ###

        ego_speed, distance_to_lead, relative_speed = self._get_obs()
        acceleration = self.action_mapping.get(action, 0)

        # velocity transition
        next_ego_speed = np.clip(ego_speed + acceleration * self.delta_t, 0, self.max_speed)

        # gap transition
        if lead_speed:
            relative_speed = lead_speed - ego_speed
            next_distance_gap = distance_to_lead + (relative_speed*self.delta_t) - (0.5*action*self.delta_t**2) #ego(v) - lead(v)
            next_relative_speed = lead_speed - next_ego_speed
        else:
            next_distance_gap = distance_to_lead + (relative_speed*self.delta_t) - (0.5*action*self.delta_t**2) #ego(v) - lead(v)
            # relative speed transition
            next_relative_speed = self.simulator.smooth_relative_speed(relative_speed)

        # Update state
        self.state = np.array([
            next_ego_speed,
            next_distance_gap,
            next_relative_speed,
        ], dtype=np.float32)

        terminated = False
        truncated = False
        reward = 0

        # Episode termination criterion -> crash
        if next_distance_gap < 0.5 or next_distance_gap > self.max_distance:
            terminated = True
            reward = -1

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _get_obs(self):
        """Return the current observation (state)."""
        return self.state

    def _get_info(self):
        """Return additional information about the current state."""
        return {
            "current_ego_speed": self.state[0],
            "distance_to_lead": self.state[1],
            "relative_speed": self.state[2],
        }

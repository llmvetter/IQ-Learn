from typing import Optional, Any

import gym
import numpy as np
import gymnasium as gym

class CarFollowingEnv(gym.Env):
    def __init__(
            self,
            max_speed=30,
            max_distance=100,
            actions: list[float]= [-0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2],
            delta_t=0.1,
    ) -> None:
        # Environment parameters
        super().__init__()
        self.max_speed = max_speed
        self.max_distance = max_distance
        self.delta_t = delta_t

        self.action_space = gym.spaces.Discrete(len(actions))
        self.action_mapping = {i: actions[i] for i in range(len(actions))}

        # state: [ego speed, distance to lead vehicle]
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0]),
            high=np.array([self.max_speed, self.max_distance]),
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
        ], dtype=np.float32)
        return self._get_obs(), self._get_info()

    def step(
            self,
            action: int,
            lead_speed: Optional[float] = None,
    ) -> None:
        """Take an action and return the next state, reward, done flag, and additional info."""

        ego_speed, distance_to_lead = self._get_obs()
        acceleration = self.action_mapping.get(action, 0)

        # velocity transition
        ego_speed = np.clip(ego_speed + acceleration * self.delta_t, 0, self.max_speed)

        # Gap transition
        if lead_speed:
            relative_speed = lead_speed - ego_speed
            next_distance = distance_to_lead - (relative_speed*self.delta_t) - (0.5*action*self.delta_t**2) #ego(v) - lead(v)
        else:
            relative_speed = np.random.normal(0, 0.8628)
            distance_next_mean = distance_to_lead - (relative_speed*self.delta_t) - (0.5*action*self.delta_t**2) #ego(v) - lead(v)
            #next_distance = np.random.normal(loc=distance_next_mean, scale=0.5)
        next_distance = np.clip(distance_next_mean , 0, self.max_distance)

        # Update state
        self.state = np.array([ego_speed, next_distance], dtype=np.float32)


        terminated = False
        truncated = False
        reward = 0

        # Episode termination criteria
        if next_distance < 0.5 or next_distance == self.max_distance:
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
        }

import gym
import numpy as np
from gym import spaces

class CarFollowingEnv(gym.Env):
    def __init__(
            self,
            max_speed=30,
            max_distance=100,
            max_acceleration=1.5,
            delta_t=0.1,
    ) -> None:
        super().__init__()

        # Environment parameters
        self.max_speed = max_speed
        self.max_distance = max_distance
        self.max_acceleration = max_acceleration
        self.delta_t = delta_t
        self.action_space = spaces.Discrete(5)

        # state: [ego speed, distance to lead vehicle]
        self.observation_space = spaces.Box(
            low=np.array([0, 0]),
            high=np.array([self.max_speed, self.max_distance]),
            dtype=np.float32
        )

        self.action_mapping = {
            0: -self.max_acceleration,
            1: -self.max_acceleration / 2,
            2: 0,
            3: self.max_acceleration / 2,
            4: self.max_acceleration
        }

        self.state = None

    def reset(self):
        """Reset the environment to an initial state."""
        self.state = np.array([
            np.random.uniform(0, self.max_speed),  # ego speed
            np.random.uniform(10, self.max_distance/2),  # distance to lead vehicle
        ])
        return self.state

    def step(self, action: int):
        """Take an action and return the next state, reward, done flag, and additional info."""

        ego_speed, distance_to_lead = self.state
        acceleration = self.action_mapping.get(action, 0)
        ego_speed = np.clip(ego_speed + acceleration * self.delta_t, 0, self.max_speed)

        relative_speed = np.random.normal(0, 0.8628)
        distance_next_mean = distance_to_lead - (relative_speed*self.delta_t) -(0.5*action*self.delta_t**2) #ego(v) - lead(v)
        next_distance = np.random.normal(loc=distance_next_mean, scale=0.5)
        next_distance = np.clip(next_distance, 0, self.max_distance)
        self.state = np.array([ego_speed, next_distance])
        
        reward = 0
        done = 0

        # maybe add high negative reward for crash scenario and flag as done
        if next_distance < 0.2:
            reward = -100
            done = 1

        return self.state, reward, done, {}

    def _get_obs(self):
        """Return the current observation (state)."""
        return self.state

    def _get_info(self):
        """Return additional information about the current state."""
        return {
            "current_ego_speed": self.state[0],
            "distance_to_lead": self.state[1],
        }

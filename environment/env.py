import gym
import numpy as np
from gym import spaces

class CarFollowingEnv(gym.Env):
    def __init__(self, max_speed=30, max_distance=100, max_acceleration=3, delta_t=0.1):
        super().__init__()

        # Environment parameters
        self.max_speed = max_speed
        self.max_distance = max_distance
        self.max_acceleration = max_acceleration
        self.delta_t = delta_t

        # Define action space: acceleration/deceleration
        self.action_space = spaces.Discrete(5)

        # Define observation space: [ego speed, distance to lead vehicle, lead speed, desired distance]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0]),
            high=np.array([self.max_speed, self.max_distance, self.max_speed, self.max_distance]),
            dtype=np.float32
        )

        self.action_mapping = {
            0: -self.max_acceleration,  # Max Deceleration
            1: -self.max_acceleration / 2,  # Medium Deceleration
            2: 0,  # No Change (Neutral)
            3: self.max_acceleration / 2,  # Medium Acceleration
            4: self.max_acceleration  # Max Acceleration
        }

        # Initialize state variable
        self.state = None

    def reset(self):
        """Reset the environment to an initial state."""
        self.state = np.array([
            np.random.uniform(0, self.max_speed),  # Ego vehicle speed
            np.random.uniform(10, self.max_distance),  # Distance to lead vehicle
            np.random.uniform(0, self.max_speed),  # Lead vehicle speed
            np.random.uniform(10, self.max_distance)  # Desired following distance
        ])
        return self.state

    def step(self, action):
        """Take an action and return the next state, reward, done flag, and additional info."""
        ego_speed, distance_to_lead, lead_speed, desired_distance = self.state
        
        # Apply acceleration/deceleration from action
        acceleration = self.action_mapping.get(action, 0)
        
        # Update ego vehicle speed
        ego_speed = np.clip(ego_speed + acceleration * self.delta_t, 0, self.max_speed)
        
        # Update lead vehicle speed (simple model with noise)
        lead_speed += np.random.normal(0, 0.1)
        
        # Calculate new distance to lead vehicle based on speeds
        distance_to_lead += (lead_speed - ego_speed) * self.delta_t
        
        # Ensure distance does not exceed maximum or fall below zero
        distance_to_lead = np.clip(distance_to_lead, 0, self.max_distance)

        # Update state with new values
        self.state = np.array([ego_speed, distance_to_lead, lead_speed, desired_distance])

        # Calculate reward based on how close the ego vehicle is to the desired following distance
        reward = -abs(distance_to_lead - desired_distance) - 0.1 * abs(acceleration)

        done = False  # Continuous task; no terminal state

        return self.state, reward, done, {}

    def _get_obs(self):
        """Return the current observation (state)."""
        return self.state

    def _get_info(self):
        """Return additional information about the current state."""
        return {
            "current_ego_speed": self.state[0],
            "distance_to_lead": self.state[1],
            "lead_vehicle_speed": self.state[2],
            "desired_distance": self.state[3]
        }

import gym
import numpy as np

class OneHotFrozenLakeWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.n_states = env.observation_space.n  # Total number of states
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.n_states,), dtype=np.float32)

    def observation(self, state):
        """Convert the integer state to a one-hot encoded vector."""
        one_hot = np.zeros(self.n_states, dtype=np.float32)
        one_hot[state] = 1.0
        return one_hot

# Create wrapped environment
env = gym.make("FrozenLake-v1", is_slippery=False)  # You can set is_slippery=True for stochastic behavior
env = OneHotFrozenLakeWrapper(env)

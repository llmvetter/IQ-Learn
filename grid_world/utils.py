import gymnasium as gym
from gymnasium import Env

def make_env() -> Env:
    return gym.make("FrozenLake-v1", is_slippery=True, map_name="8x8")

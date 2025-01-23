from environment.env import CarFollowingEnv

def make_env():

    env = CarFollowingEnv(
        max_speed=30,
        max_distance=100,
        max_acceleration=3,
        delta_t=0.1,
    )
    
    return env
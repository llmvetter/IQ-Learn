import random
from collections import deque

import numpy as np
import torch

from make_envs import make_env
from dataset.memory import Memory
from agent import make_agent
from omegaconf import OmegaConf

# Set basic configurations
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPLAY_MEMORY = 10000
INITIAL_MEMORY = 1000
EPISODE_STEPS = 200
LEARN_STEPS = 10000
EVAL_INTERVAL = 1000
LOG_INTERVAL = 100


def main():

    # set seeds
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    env = make_env()
    config = OmegaConf.load('C:/Users/lenna/Documents/IRL/IQ-Learn/conf/config.yaml')

    agent = make_agent(
        env=env,
        args=config,
    )
    memory_replay = Memory(REPLAY_MEMORY, SEED)

    steps = 0
    learn_steps = 0
    begin_learn = False

     # Track rewards
    rewards_window = deque(maxlen=100)  # For tracking recent rewards

    print('Starting training ....')

    for epoch in range(1,10):
        state = env.reset()
        episode_reward = 0
        done = False

        for episode_step in range(EPISODE_STEPS):

            if steps < range(EPISODE_STEPS):
                action = env.action_space.sample()
            else:
                action = agent.choose_action(state, sample=True)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            steps += 1
            memory_replay.add((state, next_state, action, reward, done))

            if memory_replay.size() > INITIAL_MEMORY:
                # Start learning
                if not begin_learn:
                    print('Learn begins!')
                    begin_learn = True

                learn_steps += 1
                losses = agent.update(memory_replay)

                if learn_steps % LOG_INTERVAL == 0:
                    print(f"Step {learn_steps}: Losses {losses}")

                if learn_steps >= LEARN_STEPS:
                    print("Finished training!")
                    return

            if done:
                break
            state = next_state

        rewards_window.append(episode_reward)
        print('TRAIN\tEp {}\tAverage reward: {:.2f}\t'.format(epoch, np.mean(rewards_window)))


if __name__ == "__main__":
    main()

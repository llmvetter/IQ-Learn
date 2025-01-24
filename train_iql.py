import random

import torch
import numpy as np
from omegaconf import OmegaConf

from make_envs import make_env
from agent import make_agent
from dataset.memory import Memory
from dataset.preprocessor import Preprocessor
from dataset.expert_dataset import ExpertDataset


def main():

    args = OmegaConf.load('C:/Users/lenna/Documents/IRL/IQ-Learn/conf/config.yaml')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    REPLAY_MEMORY = int(args.env.replay_mem)
    INITIAL_MEMORY = int(args.env.initial_mem)
    EPISODE_STEPS = int(args.env.eps_steps)
    EPISODE_WINDOW = int(args.env.eps_window)
    LEARN_STEPS = int(args.env.learn_steps)
    INITIAL_STATES = 128

    env = make_env()

    args.agent.critic_cfg.obs_dim = env.observation_space.shape[0]
    args.agent.critic_cfg.action_dim = env.action_space.n

    agent = make_agent(
        env=env,
        args=args,
    )

    preprocessor = Preprocessor(env)
    expert_trajectories = preprocessor.load("C:/Users/lenna/Documents/IRL/data/TrajData_Punzo_Napoli/drivetest1.FCdata")
    ex_data = ExpertDataset(expert_trajectories=expert_trajectories)
    expert_memory_replay = Memory(REPLAY_MEMORY//2, args.seed)
    expert_memory_replay.load(ex_data)

    print(f'--> Expert memory size: {expert_memory_replay.size()}')

    online_memory_replay = Memory(REPLAY_MEMORY//2, args.seed+1)

    steps = 0
    learn_steps = 0
    begin_learn = False
    episode_reward = 0

    state_0 = [env.reset()] * INITIAL_STATES
    state_0 = torch.FloatTensor(np.array(state_0))

    for epoch in range(args.train.epochs):
        state = env.reset()
        episode_rewards = 0
        done = False

        for episode_step in range(EPISODE_STEPS):

            if steps < args.num_seed_steps:
                action = env.action_space.sample()
            else:
                action = agent.choose_action(state, sample=True)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            steps += 1
            online_memory_replay.add((state, next_state, action, reward, done))

            if online_memory_replay.size() > INITIAL_MEMORY:
                if begin_learn is False:
                    print('Learn begins!')
                    begin_learn = True

                learn_steps += 1
                if learn_steps == LEARN_STEPS:
                    print('Finished!')
                    return
            


    
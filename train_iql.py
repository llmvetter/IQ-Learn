import random

import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from environment.make_envs import make_env
from agent.softq import SoftQ
from dataset.memory import Memory
from dataset.preprocessor.preprocessor import NapoliPreprocessor
from dataset.expert_dataset import ExpertDataset
from utils.utils import get_concat_samples

def main():
    save_path = "C:/Users/lenna/Documents/IRL/IQ-Learn/results/trained_agent.pkl"
    args = OmegaConf.load('C:/Users/lenna/Documents/IRL/IQ-Learn/config/config.yaml')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    REPLAY_MEMORY = int(args.env.replay_mem)
    INITIAL_MEMORY = int(args.env.initial_mem)
    EPISODE_STEPS = int(args.env.eps_steps)
    LEARN_STEPS = int(args.env.learn_steps)
    INITIAL_STATES = 128

    env = make_env()

    args.agent.critic_cfg.obs_dim = env.observation_space.shape[0]
    args.agent.critic_cfg.action_dim = env.action_space.n

    agent = SoftQ(
        batch_size=args.train.batch,
        args=args,
    )

    preprocessor = NapoliPreprocessor(env)
    expert_trajectories = preprocessor.preprocess()
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
    loss_tracker = pd.DataFrame(columns=["epoch", "learn_step", "softq_loss", "value_loss", "chi2_loss", "total_loss"])

    for epoch in range(args.train.epochs):
        state = env.reset()
        done = False

        for _ in range(EPISODE_STEPS):

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
                    print('Starting training loop!')
                    begin_learn = True

                learn_steps += 1
                if learn_steps == LEARN_STEPS:
                    print('Finished training loop!')
                    return

                losses = agent.iq_update(
                    policy_buffer = online_memory_replay,
                    expert_buffer = expert_memory_replay,
                    step = learn_steps,
                )
                if learn_steps % 1000 == 0:
                    print(f"total_loss {losses}, learn step: {learn_steps}")
                    loss_tracker = pd.concat([loss_tracker, pd.DataFrame([{
                        "epoch": epoch,
                        "learn_step": learn_steps,
                        "softq_loss": losses.get("softq_loss", 0),
                        "value_loss": losses.get("value_loss", 0),
                        "chi2_loss": losses.get("chi2_loss", 0),
                        "total_loss": losses.get("total_loss", 0)
                    }])], ignore_index=True)

            if done:
                break
            state = next_state
    
        print(f'train/epoch: {epoch}')
    
    #safe agent
    with open(save_path, "wb") as f:
        pickle.dump(agent, f)
    print(f'Agent has been saved at: {save_path}')

    # Plot the losses
    plt.figure(figsize=(12, 6))
    plt.plot(loss_tracker["learn_step"], loss_tracker["softq_loss"], label="SoftQ Loss", alpha=0.7)
    plt.plot(loss_tracker["learn_step"], loss_tracker["value_loss"], label="Value Loss", alpha=0.7)
    plt.plot(loss_tracker["learn_step"], loss_tracker["chi2_loss"], label="Chi2 Loss", alpha=0.7)
    plt.plot(loss_tracker["learn_step"], loss_tracker["total_loss"], label="Total Loss", alpha=0.9, linewidth=2)

    # Add labels, legend, and title
    plt.xlabel("Learn Steps")
    plt.ylabel("Loss")
    plt.title("Loss Trends Over Training")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

# Minimal IQ-Learn objective
def iq_learn_update(self, policy_batch, expert_batch):
    args = self.args
    policy_obs, policy_next_obs, policy_action, policy_reward, policy_done = policy_batch
    expert_obs, expert_next_obs, expert_action, expert_reward, expert_done = expert_batch

    if args.only_expert_states:
        expert_batch = expert_obs, expert_next_obs, policy_action, expert_reward, expert_done

    obs, next_obs, action, reward, done, is_expert = get_concat_samples(
        policy_batch, expert_batch, args)

    ######
    # IQ-Learn minimal implementation with X^2 divergence (~15 lines)
    # Calculate 1st term of loss: -E_(ρ_expert)[Q(s, a) - γV(s')]
    current_Q = self.critic(obs, action)
    y = (1 - done) * self.gamma * self.getV(next_obs)
    if args.train.use_target:
        with torch.no_grad():
            y = (1 - done) * self.gamma * self.get_targetV(next_obs)

    reward = (current_Q - y)[is_expert]
    loss = -(reward).mean()

    # 2nd term for our loss (use expert and policy states): E_(ρ)[Q(s,a) - γV(s')]
    value_loss = (self.getV(obs) - y).mean()
    loss += value_loss

    # Use χ2 divergence (adds a extra term to the loss)
    chi2_loss = 1/(4 * args.method.alpha) * (reward**2).mean()
    loss += chi2_loss
    ######

    self.critic_optimizer.zero_grad()
    loss.backward()
    self.critic_optimizer.step()
    return loss

if __name__ == "__main__":
    main()
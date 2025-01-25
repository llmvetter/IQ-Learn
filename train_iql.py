import random
import types

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from make_envs import make_env
from agent import make_agent
from dataset.memory import Memory
from dataset.preprocessor import Preprocessor
from dataset.expert_dataset import ExpertDataset
from utils.utils import average_dicts, get_concat_samples, soft_update, hard_update
from iq import iq_loss

def main():

    args = OmegaConf.load('C:/Users/lenna/Documents/IRL/IQ-Learn/conf/config.yaml')

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
                    print('Learn begins!')
                    begin_learn = True

                learn_steps += 1
                if learn_steps == LEARN_STEPS:
                    print('Finished!')
                    return
            
                agent.iq_update = types.MethodType(iq_update, agent)
                agent.iq_update_critic = types.MethodType(iq_update_critic, agent)
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

def iq_update_critic(self, policy_batch, expert_batch, step):
    args = self.args
    policy_obs, policy_next_obs, policy_action, policy_reward, policy_done = policy_batch
    expert_obs, expert_next_obs, expert_action, expert_reward, expert_done = expert_batch

    if args.only_expert_states:
        # Use policy actions instead of experts actions for IL with only observations
        expert_batch = expert_obs, expert_next_obs, policy_action, expert_reward, expert_done

    batch = get_concat_samples(policy_batch, expert_batch, args)
    obs, next_obs, action = batch[0:3]

    agent = self
    current_V = self.getV(obs)
    if args.train.use_target:
        with torch.no_grad():
            next_V = self.get_targetV(next_obs)
    else:
        next_V = self.getV(next_obs)

    if "DoubleQ" in self.args.q_net._target_:
        current_Q1, current_Q2 = self.critic(obs, action, both=True)
        q1_loss, loss_dict1 = iq_loss(agent, current_Q1, current_V, next_V, batch)
        q2_loss, loss_dict2 = iq_loss(agent, current_Q2, current_V, next_V, batch)
        critic_loss = 1/2 * (q1_loss + q2_loss)
        # merge loss dicts
        loss_dict = average_dicts(loss_dict1, loss_dict2)
    else:
        current_Q = self.critic(obs, action)
        critic_loss, loss_dict = iq_loss(agent, current_Q, current_V, next_V, batch)

    # Optimize the critic
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    # step critic
    self.critic_optimizer.step()
    return loss_dict

def iq_update(self, policy_buffer, expert_buffer, step):
    policy_batch = policy_buffer.get_samples(self.batch_size, self.device)
    expert_batch = expert_buffer.get_samples(self.batch_size, self.device)

    losses = self.iq_update_critic(policy_batch, expert_batch, step)

    if self.actor and step % self.actor_update_frequency == 0:
        if not self.args.agent.vdice_actor:

            if self.args.offline:
                obs = expert_batch[0]
            else:
                # Use both policy and expert observations
                obs = torch.cat([policy_batch[0], expert_batch[0]], dim=0)

            if self.args.num_actor_updates:
                for i in range(self.args.num_actor_updates):
                    actor_alpha_losses = self.update_actor_and_alpha(obs, step)

            losses.update(actor_alpha_losses)

    if step % self.critic_target_update_frequency == 0:
        if self.args.train.soft_update:
            soft_update(self.critic_net, self.critic_target_net,
                        self.critic_tau)
        else:
            hard_update(self.critic_net, self.critic_target_net)
    return losses
    

if __name__ == "__main__":
    main()
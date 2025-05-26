from typing import Any
from tqdm import tqdm

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

from src.agent.softq import SoftQ
from src.environment.env import CarFollowingEnv


def weighted_softmax(x, weights):
    x = x - torch.max(x, dim=0)[0]
    return weights * torch.exp(x) / torch.sum(
        weights * torch.exp(x), dim=0, keepdim=True)


def soft_update(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def hard_update(source, target):
    for param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(param.data)


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def get_concat_samples(policy_batch, expert_batch, args):
    online_batch_state, online_batch_next_state, online_batch_action, online_batch_reward, online_batch_done = policy_batch

    expert_batch_state, expert_batch_next_state, expert_batch_action, expert_batch_reward, expert_batch_done = expert_batch

    if args.method.type == "sqil":
        # convert policy reward to 0
        online_batch_reward = torch.zeros_like(online_batch_reward)
        # convert expert reward to 1
        expert_batch_reward = torch.ones_like(expert_batch_reward)

    batch_state = torch.cat([online_batch_state, expert_batch_state], dim=0)
    batch_next_state = torch.cat(
        [online_batch_next_state, expert_batch_next_state], dim=0)
    batch_action = torch.cat([online_batch_action, expert_batch_action], dim=0)
    batch_reward = torch.cat([online_batch_reward, expert_batch_reward], dim=0)
    batch_done = torch.cat([online_batch_done, expert_batch_done], dim=0)
    is_expert = torch.cat([torch.zeros_like(online_batch_reward, dtype=torch.bool),
                           torch.ones_like(expert_batch_reward, dtype=torch.bool)], dim=0)

    return batch_state, batch_next_state, batch_action, batch_reward, batch_done, is_expert

def average_dicts(dict1, dict2):
    return {key: 1/2 * (dict1.get(key, 0) + dict2.get(key, 0))
                     for key in set(dict1) | set(dict2)}

def deep_merge(
        d1: dict[str, Any],
        d2: dict[str, Any],
) -> dict[str, Any]:

    merged = dict(d1)
    for key, value in d2.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged

def reward_heatmap(
        agent: SoftQ,
        env: CarFollowingEnv,
        grid_granularity: float=0.5,
        relative_speed: float=1.0,

) -> None:

    v_space = np.arange(0, env.max_speed, grid_granularity)
    g_space = np.arange(0, env.max_distance, grid_granularity)
    rel_space = relative_speed

    V, G = np.meshgrid(v_space, g_space, indexing='ij')
    state_space = np.stack([V.ravel(), G.ravel(), np.full(V.size, rel_space)], axis=1)

    obs = []
    obs_action = []
    next_obs = []
    dones = []

    for state in tqdm(state_space):
        env.reset()
        env.state = state
        action = agent.choose_action(state, sample=False)
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        dones.append(1 if terminated else 0)
        obs.append(torch.FloatTensor(state).unsqueeze(0))
        obs_action.append(action)
        next_obs.append(torch.FloatTensor(next_state).unsqueeze(0))

    rewards = []
    with torch.no_grad():
        for i in tqdm(range(len(obs))):
            q_values = agent.q_net(obs[i])
            irl_reward = torch.max(q_values)
            # q = q_values[0, obs_action[i]]

            # next_v = agent.getV(next_obs[i])
            # y = (1 - dones[i]) * agent.gamma * next_v
            # irl_reward = q - y
            rewards.append(irl_reward.item())
    rewards = np.array(rewards)
    rewards[rewards < 0] = 0
    rewards = np.array(rewards).reshape(len(v_space), len(g_space))

    # ---- 2D Heatmap ----
    _, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        state_space[:, 1],
        state_space[:, 0],
        c=rewards,
        cmap='viridis',
    )
    plt.colorbar(scatter, label='Reward')
    ax.set_xlabel('Distance Gap g (m)')
    ax.set_ylabel('Velocity v in m/s')
    ax.set_title('Reward heatmap over State Space')
    plt.show()

from omegaconf import DictConfig, OmegaConf
import torch
from itertools import count
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import matplotlib

from agent import make_agent
from make_envs import make_env

matplotlib.use('Agg')



def main():
    args = OmegaConf.load('C:/Users/lenna/Documents/IRL/IQ-Learn/conf/config.yaml')

    GAMMA = args.gamma

    env = make_env()
    args.agent.critic_cfg.obs_dim = env.observation_space.shape[0]
    args.agent.critic_cfg.action_dim = env.action_space.n

    agent = make_agent(
        env=env,
        args=args,
    )

    policy_file = "C:/Users/lenna/Documents/IRL/IQ-Learn/results/trained_agent.pkl"
    print(f'Loading policy from: {policy_file}')
    with open(policy_file, "rb") as f:
        agent = pickle.load(f)

    visualize_reward(agent, env, args)

    for epoch in count():
        state = env.reset()
        episode_irl_reward = 0

        for _ in 1000:

            action = agent.choose_action(state, sample=False)
            next_state, _, done, _ = env.step(action)

            # Get sqil reward
            with torch.no_grad():
                q = agent.infer_q(state, action)
                next_v = agent.infer_v(next_state)
                y = (1 - done) * GAMMA * next_v
                irl_reward = q - y

            episode_irl_reward += irl_reward.item()
            
            if done:
                break
            state = next_state

        print('Ep {}\tMoving Soft Q average score: {:.2f}\t'.format(epoch, episode_irl_reward))


def visualize_reward(agent, env, args):

    env = make_env()

    grid_size_speed = 1.0  # Resolution for ego speed
    grid_size_distance = 1.0  # Resolution for distance to lead vehicle

    rescale_speed = 1.0 / grid_size_speed
    rescale_distance = 1.0 / grid_size_distance

    boundary_speed_low = 0
    boundary_speed_high = env.max_speed

    boundary_distance_low = 0
    boundary_distance_high = env.max_distance

    num_speed = 0
    for ego_speed in np.arange(boundary_speed_low, boundary_speed_high, grid_size_speed):
            num_speed += 1
            num_distance = 0
            for distance_to_lead in np.arange(boundary_distance_low, boundary_distance_high, grid_size_distance):
                num_distance += 1

                # Define the current state in the grid
                state = np.array([ego_speed, distance_to_lead])
                env.state = state  # Set environment state
                
                # Choose an action and compute the next state
                action = agent.choose_action(state, sample=False)
                next_state, reward, done, _ = env.step(action)

                obs_batch.append(state)
                obs_action.append(action)
                next_obs_batch.append(next_state)

            # Convert batches to numpy arrays
            obs_batch = np.array(obs_batch)
            next_obs_batch = np.array(next_obs_batch)
            obs_action = np.array(obs_action)

            # Get IRL rewards using the agent
            with torch.no_grad():
                state_tensor = torch.FloatTensor(obs_batch)
                action_tensor = torch.FloatTensor(obs_action)
                next_state_tensor = torch.FloatTensor(next_obs_batch)

                # Calculate Q-values and IRL rewards
                q = agent.critic(state_tensor, action_tensor)  # Q(s, a)
                next_v = agent.getV(next_state_tensor)  # V(s')
                y = (1 - done) * args.gamma * next_v  # Bellman target
                irl_reward = q - y
                irl_reward = -irl_reward.cpu().numpy()

            # Reshape and visualize rewards
            score = irl_reward.reshape([num_speed, num_distance])
            ax = sns.heatmap(score, cmap="YlGnBu_r", xticklabels=False, yticklabels=False)

            # Highlight specific points, e.g., ideal distances
            ax.scatter((env.max_speed / 2) * rescale_speed, 
                    (env.max_distance / 2) * rescale_distance, 
                    marker='*', s=150, c='r', edgecolors='k', linewidths=0.5)

            ax.invert_yaxis()
            plt.axis('off')
            plt.close()

if __name__ == '__main__':
    main()
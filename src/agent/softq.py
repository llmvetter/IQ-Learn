from typing import Any
import numpy as np
import pickle
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical
from omegaconf import OmegaConf


class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, args) -> None:
        super().__init__()
        self.args = args
        self.fc1 = nn.Linear(obs_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SoftQ(object):
    def __init__(self, args, batch_size: int = 32):
        self.gamma = args.gamma
        self.args = args
        self.batch_size = batch_size
        self.target_update_frequency = self.args.agent.target_update_frequency
        self.log_alpha = torch.tensor(np.log(self.args.agent.init_temp))
        self.q_net = SoftQNetwork(
            obs_dim=args.agent.critic_cfg.obs_dim,
            action_dim=args.agent.critic_cfg.action_dim,
            args=args,
        )
        self.target_net = SoftQNetwork(
            obs_dim=args.agent.critic_cfg.obs_dim,
            action_dim=args.agent.critic_cfg.action_dim,
            args=args,
        )
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.critic_optimizer = Adam(
            self.q_net.parameters(),
            lr=self.args.agent.critic_lr,
            betas=self.args.agent.critic_betas,
        )
        self.train()
        self.target_net.train()

    def train(self, training=True) -> nn.Module:
        self.training = training
        self.q_net.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @property
    def critic_net(self):
        return self.q_net

    @property
    def critic_target_net(self):
        return self.target_net

    def choose_action(
            self,
            state: torch.FloatTensor,
            sample=False
    ) -> torch.Tensor:

        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q = self.q_net(state)
            dist = F.softmax(q/self.alpha, dim=1)
            if sample:
                dist = Categorical(dist)
                action = dist.sample()
            else:
                action = torch.argmax(dist, dim=1)

        return action.detach().numpy()[0]

    def getV(
            self,
            obs: torch.Tensor
    ) -> torch.Tensor:
        q = self.q_net(obs)
        v = self.alpha * torch.logsumexp(q/self.alpha, dim=1, keepdim=True)
        return v
    
    def get_targetV(
            self,
            obs: torch.Tensor,
    ) -> torch.Tensor:
        q = self.target_net(obs)
        target_v = self.alpha * torch.logsumexp(q/self.alpha, dim=1, keepdim=True)
        return target_v

    def critic(
            self,
            obs: torch.Tensor,
            action: torch.Tensor,
    ) -> torch.Tensor:
        q = self.q_net(obs)
        return q.gather(1, action.long())

    def q_learn_update(
            self,
            obs: torch.Tensor,
            action: torch.Tensor,
            reward: torch.Tensor,
            next_obs: torch.Tensor,
            done: bool,
    ) -> dict[str, float]:
        # Q Learning update of critic network using mse loss
        with torch.no_grad():
            next_v = self.get_targetV(next_obs)
            y = reward + (1 - done) * self.gamma * next_v

        critic_loss = F.mse_loss(self.critic(obs, action), y)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return {'loss/critic': critic_loss.item()}

    # Minimal IQ-Learn objective
    def iq_learn_update(
            self,
            policy_batch: torch.FloatTensor,
            expert_batch: torch.FloatTensor,
    ) -> tuple[Any]:

        # Inverse Q Learning update of critic network
        obs, next_obs, action, reward, done, is_expert = get_concat_samples(
            policy_batch, expert_batch)

        ######
        # IQ-Learn minimal implementation with X^2 divergence
        # Calculate 1st term of loss: -E_(ρ_expert)[Q(s, a) - γV(s')]
        # Uses expert states only

        # Q(s,a)
        current_Q = self.critic(obs, action)

        # V(s')
        if self.target_update_frequency:
            next_v = self.get_targetV(next_obs)
        else:
            next_v = self.getV(next_obs)

        # γV(s'), will be 0 if done
        y = (1 - done.float()) * self.gamma * next_v

        # E_(ρ_expert)[Q(s, a) - γV(s')], non expert states are squeezed out
        reward = (current_Q - y)[is_expert.squeeze(-1)]

        # -E_(ρ_expert)[Q(s, a) - γV(s')]
        # batch computation, so we take the reward mean over all states
        loss_1 = -(reward).mean()

        # 2nd term for our loss (use expert and policy states): E_(ρ)[Q(s,a) - γV(s')]
        value_loss = (self.getV(obs) - y).mean()

        # Use χ2 divergence (adds a extra term to the loss)
        # Higher alpha leads to more regularization but slower learning
        chi2_loss = 1/(4 * self.args.method.alpha) * (reward**2).mean()
        #####

        total_loss = loss_1 + value_loss + chi2_loss
        #optimize critic
        self.critic_optimizer.zero_grad()
        total_loss.backward()
        #step critic
        self.critic_optimizer.step()

        return total_loss, loss_1, value_loss, chi2_loss
    

    def infer_q(self, state, action):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = torch.FloatTensor([action]).unsqueeze(0)

        with torch.no_grad():
            q = self.critic(state, action)
        return q.squeeze(0).numpy()

    def infer_v(self, state):

        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            v = self.getV(state).squeeze()
        return v.numpy()
    

def get_concat_samples(
        policy_sample: tuple[Any],
        expert_sample: tuple[Any],
) -> tuple[Any]:

    online_s, online_next_s, online_a, online_r, online_done = policy_sample
    expert_s, expert_next_s, expert_a, expert_r, expert_done = expert_sample

    batch_state = torch.cat([online_s, expert_s], dim=0)
    batch_next_state = torch.cat([online_next_s, expert_next_s], dim=0)
    batch_action = torch.cat([online_a, expert_a], dim=0)
    batch_reward = torch.cat([online_r, expert_r], dim=0)
    batch_done = torch.cat([online_done, expert_done], dim=0)
    is_expert = torch.cat([torch.zeros_like(online_r, dtype=torch.bool),
                           torch.ones_like(expert_r, dtype=torch.bool)], dim=0)

    return batch_state, batch_next_state, batch_action, batch_reward, batch_done, is_expert

def load_model(
        model_pickle: str,
        model_config_path: str,
) -> SoftQ:
    config = OmegaConf.load(model_config_path)
    with open(model_pickle, "rb") as f:
        saved_data = pickle.load(f)
        # Restore model
    soft_q = SoftQ(config)  # Recreate object with same args
    soft_q.q_net.load_state_dict(saved_data["q_net"])
    soft_q.target_net.load_state_dict(saved_data["target_net"])
    soft_q.critic_optimizer.load_state_dict(saved_data["critic_optimizer"])
    soft_q.log_alpha = saved_data["log_alpha"]  # Restore log_alpha

    return soft_q


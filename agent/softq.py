import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Categorical

from agent.softq_models import SimpleQNetwork
from utils.utils import average_dicts, get_concat_samples
from iq import iq_loss

class SoftQ(object):
    def __init__(self, batch_size, args):
        self.gamma = args.gamma
        self.batch_size = batch_size
        self.device = torch.device(args.device)
        self.args = args
        agent_cfg = args.agent
        self.actor = None
        self.critic_tau = agent_cfg.critic_tau

        self.critic_target_update_frequency = agent_cfg.critic_target_update_frequency
        self.log_alpha = torch.tensor(np.log(agent_cfg.init_temp)).to(self.device)
        self.q_net = SimpleQNetwork(
            obs_dim=args.agent.critic_cfg.obs_dim,
            action_dim=args.agent.critic_cfg.action_dim,
            args=args,
        )
        self.target_net = SimpleQNetwork(
            obs_dim=args.agent.critic_cfg.obs_dim,
            action_dim=args.agent.critic_cfg.action_dim,
            args=args,
        )
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.critic_optimizer = Adam(self.q_net.parameters(), lr=agent_cfg.critic_lr,
                                     betas=agent_cfg.critic_betas)
        self.train()
        self.target_net.train()

    def train(self, training=True):
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

    def choose_action(self, state, sample=False):

        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.q_net(state)
            dist = F.softmax(q/self.alpha, dim=1)
            # if sample:
            dist = Categorical(dist)
            action = dist.sample()  # if sample else dist.mean
            # else:
            #     action = torch.argmax(dist, dim=1)

        return action.detach().cpu().numpy()[0]

    def getV(self, obs):
        q = self.q_net(obs)
        v = self.alpha * \
            torch.logsumexp(q/self.alpha, dim=1, keepdim=True)
        return v

    def critic(self, obs, action, both=False):
        q = self.q_net(obs, both)
        if isinstance(q, tuple) and both:
            q1, q2 = q
            critic1 = q1.gather(1, action.long())
            critic2 = q2.gather(1, action.long())
            return critic1, critic2

        return q.gather(1, action.long())

    def get_targetV(self, obs):
        q = self.target_net(obs)
        target_v = self.alpha * \
            torch.logsumexp(q/self.alpha, dim=1, keepdim=True)
        return target_v

    def update(self, replay_buffer, step):
        obs, next_obs, action, reward, done = replay_buffer.get_samples(
            self.batch_size, self.device)

        losses = self.update_critic(obs, action, reward, next_obs, done, step)

        if step % self.critic_target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return losses

    def update_critic(self, obs, action, reward, next_obs, done, step):

        with torch.no_grad():
            next_v = self.get_targetV(next_obs)
            y = reward + (1 - done) * self.gamma * next_v

        critic_loss = F.mse_loss(self.critic(obs, action), y)
        #print('train_critic/loss', critic_loss, step)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return {
            'loss/critic': critic_loss.item()}

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
        return losses
    
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

    # Save model parameters
    def save(self, path, suffix=""):
        critic_path = f"{path}{suffix}"
        # print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.q_net.state_dict(), critic_path)

    # Load model parameters
    def load(self, path, suffix=""):
        critic_path = f'{path}/{self.args.agent.name}{suffix}'
        print('Loading models from {}'.format(critic_path))
        self.q_net.load_state_dict(torch.load(critic_path, map_location=self.device))

    def infer_q(self, state, action):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        action = torch.FloatTensor([action]).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q = self.critic(state, action)
        return q.squeeze(0).cpu().numpy()

    def infer_v(self, state):

        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            v = self.getV(state).squeeze()
        return v.cpu().numpy()

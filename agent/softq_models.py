import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
from torch.autograd import Variable, grad


class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, args, device='cpu'):
        super(SoftQNetwork, self).__init__()
        self.args = args
        self.device = device
        self.tanh = nn.Tanh()

    def _forward(self, x, *args):
        return NotImplementedError

    def forward(self, x, both=False):
        if "DoubleQ" in self.args.q_net._target_:
            out = self._forward(x, both)
        else:
            out = self._forward(x)

        if getattr(self.args.method, "tanh", False):
            return self.tanh(out) * 1/(1-self.args.gamma)
        return out

    def jacobian(self, outputs, inputs):
        """Computes the jacobian of outputs with respect to inputs

        :param outputs: tensor for the output of some function
        :param inputs: tensor for the input of some function (probably a vector)
        :returns: a tensor containing the jacobian of outputs with respect to inputs
        """
        batch_size, output_dim = outputs.shape
        jacobian = []
        for i in range(output_dim):
            v = torch.zeros_like(outputs)
            v[:, i] = 1.
            dy_i_dx = grad(outputs,
                           inputs,
                           grad_outputs=v,
                           retain_graph=True,
                           create_graph=True)[0]  # shape [B, N]
            jacobian.append(dy_i_dx)

        jacobian = torch.stack(jacobian, dim=-1).requires_grad_()
        return jacobian

    def grad_pen(self, obs1, action1, obs2, action2, lambda_=1):
        expert_data = obs1
        policy_data = obs2
        batch_size = expert_data.size()[0]

        # Calculate interpolation
        if expert_data.ndim == 4:
            alpha = torch.rand(batch_size, 1, 1, 1)  # B, C, H, W input
        else:
            alpha = torch.rand(batch_size, 1)
        alpha = alpha.expand_as(expert_data).to(expert_data.device)

        interpolated = alpha * expert_data + (1 - alpha) * policy_data
        interpolated = Variable(interpolated, requires_grad=True)
        interpolated = interpolated.to(expert_data.device)

        # Calculate probability of interpolated examples
        prob_interpolated = self.forward(interpolated)
        # Calculate gradients of probabilities with respect to examples
        gradients = self.jacobian(prob_interpolated, interpolated)

        # Gradients have shape (batch_size, input_dim, output_dim)
        out_size = gradients.shape[-1]
        gradients_norm = gradients.reshape([batch_size, -1, out_size]).norm(2, dim=1)

        # Return gradient penalty
        return lambda_ * ((gradients_norm - 1) ** 2).mean()


class SimpleQNetwork(SoftQNetwork):
    def __init__(self, obs_dim, action_dim, args, device='cpu'):
        super(SimpleQNetwork, self).__init__(obs_dim, action_dim, args, device)
        self.args = args
        self.fc1 = nn.Linear(obs_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def _forward(self, x, *args):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class OfflineQNetwork(SoftQNetwork):
    def __init__(self, obs_dim, action_dim, args, device='cpu'):
        super(OfflineQNetwork, self).__init__(obs_dim, action_dim, args, device)
        self.args = args
        self.fc1 = nn.Linear(obs_dim, 64)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def _forward(self, x, *args):
        x = self.elu(self.fc1(x))
        x = self.elu(self.fc2(x))
        x = self.fc3(x)
        return x


class SimpleVNetwork(SoftQNetwork):
    def __init__(self, obs_dim, action_dim, args, device='cpu'):
        super(SimpleVNetwork, self).__init__(obs_dim, action_dim, args, device)
        self.args = args
        self.fc1 = nn.Linear(obs_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def _forward(self, x, *args):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

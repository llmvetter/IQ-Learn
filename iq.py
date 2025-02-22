"""
Copyright 2022 Div Garg. All rights reserved.

Standalone IQ-Learn algorithm. See LICENSE for licensing terms.
"""
import torch
import torch.nn.functional as F


# Full IQ-Learn objective with other divergences and options
def iq_loss(agent, current_Q, current_v, next_v, batch):
    args = agent.args
    gamma = agent.gamma
    obs, next_obs, action, env_reward, done, is_expert = batch

    loss_dict = {}

    #  calculate 1st term for IQ loss
    #  -E_(ρ_expert)[Q(s, a) - γV(s')]
    y = (1 - done) * gamma * next_v
    reward = (current_Q - y)[is_expert]
    div_type = getattr(args.method, 'div', None)

    with torch.no_grad():
        # Use different divergence functions (For χ2 divergence we instead add a third bellmann error-like term)
        if div_type == "hellinger":
            phi_grad = 1/(1+reward)**2
        elif div_type == "kl":
            # original dual form for kl divergence (sub optimal)
            phi_grad = torch.exp(-reward-1)
        elif div_type == "kl2":
            # biased dual form for kl divergence
            phi_grad = F.softmax(-reward, dim=0) * reward.shape[0]
        elif div_type == "kl_fix":
            # our proposed unbiased form for fixing kl divergence
            phi_grad = torch.exp(-reward)
        elif div_type == "js":
            # jensen–shannon
            phi_grad = torch.exp(-reward)/(2 - torch.exp(-reward))
        else:
            phi_grad = 1
    loss = -(phi_grad * reward).mean()
    loss_dict['softq_loss'] = loss.item()

    # calculate 2nd term for IQ loss, we show different sampling strategies
    if args.method.loss == "value_expert":
        # sample using only expert states (works offline)
        # E_(ρ)[Q(s,a) - γV(s')]
        value_loss = (current_v - y)[is_expert].mean()
        loss += value_loss
        loss_dict['value_loss'] = value_loss.item()

    elif args.method.loss == "value":
        # sample using expert and policy states (works online)
        # E_(ρ)[V(s) - γV(s')]
        value_loss = (current_v - y).mean()
        loss += value_loss
        loss_dict['value_loss'] = value_loss.item()

    # alternative sampling strategies for the sake of completeness but are usually suboptimal in practice
    # elif args.method.loss == "value_policy":
    #     # sample using only policy states
    #     # E_(ρ)[V(s) - γV(s')]
    #     value_loss = (current_v - y)[~is_expert].mean()
    #     loss += value_loss
    #     loss_dict['value_policy_loss'] = value_loss.item()

    # elif args.method.loss == "value_mix":
    #     # sample by weighted combination of expert and policy states
    #     # E_(ρ)[Q(s,a) - γV(s')]
    #     w = args.method.mix_coeff
    #     value_loss = (w * (current_v - y)[is_expert] +
    #                   (1-w) * (current_v - y)[~is_expert]).mean()
    #     loss += value_loss
    #     loss_dict['value_loss'] = value_loss.item()

    else:
        raise ValueError(f'This sampling method is not implemented: {args.method.type}')

    if args.method.grad_pen:
        # add a gradient penalty to loss (Wasserstein_1 metric)
        gp_loss = agent.critic_net.grad_pen(obs[is_expert.squeeze(1), ...],
                                            action[is_expert.squeeze(1), ...],
                                            obs[~is_expert.squeeze(1), ...],
                                            action[~is_expert.squeeze(1), ...],
                                            args.method.lambda_gp)
        loss_dict['gp_loss'] = gp_loss.item()
        loss += gp_loss

    if args.method.div == "chi":
        # Use χ2 divergence (calculate the regularization term for IQ loss using expert states) (works offline)
        y = (1 - done) * gamma * next_v
        reward = current_Q - y
        chi2_loss = 1/(4 * args.method.alpha) * (reward**2)[is_expert].mean()
        loss += chi2_loss
        loss_dict['chi2_loss'] = chi2_loss.item()

    if args.method.regularize:
        # Use χ2 divergence (calculate the regularization term for IQ loss using expert and policy states) (works online)
        y = (1 - done) * gamma * next_v

        reward = current_Q - y
        chi2_loss = 1/(4 * args.method.alpha) * (reward**2).mean()
        loss += chi2_loss
        loss_dict['regularize_loss'] = chi2_loss.item()

    loss_dict['total_loss'] = loss.item()
    return loss, loss_dict

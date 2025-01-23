from agent.softq import SoftQ


def make_agent(env, args):
    obs_dim = env.observation_space.shape[0]
    print('--> Using Soft-Q agent')
    action_dim = env.action_space.n
    # TODO: Simplify logic
    args.agent.obs_dim = obs_dim
    args.agent.action_dim = action_dim
    agent = SoftQ(obs_dim, action_dim, args.train.batch, args)
    
    return agent
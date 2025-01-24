from agent.softq import SoftQ


def make_agent(env, args):

    args.agent.obs_dim = env.observation_space.shape[0]
    args.agent.action_dim = env.action_space.n

    agent = SoftQ(
        batch_size=args.train.batch,
        args=args,
    )
    
    return agent
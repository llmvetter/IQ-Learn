import matplotlib.pyplot as plt
from tqdm import tqdm
from omegaconf import OmegaConf

from environment.env import CarFollowingEnv
from dataset.expert_dataset import ExpertDataset
from dataset.memory import Memory
from agent.softq import SoftQ



class Trainer():
    def __init__(
            self,
            config: OmegaConf,
            environment: CarFollowingEnv,
            expert_data: ExpertDataset,
    ) -> SoftQ:
        self.config = config
        self.env = environment
        self.data = expert_data
        # Init Actor
        if self.config.env.name == 'CarFollowing':
            self.config.agent.critic_cfg.obs_dim = self.env.observation_space.shape[0]
            self.config.agent.critic_cfg.action_dim = self.env.action_space.n.item()
        if self.config.env.name == 'FrozenLake':
            self.config.agent.critic_cfg.obs_dim = 64
            self.config.agent.critic_cfg.action_dim = 4
        self.agent = SoftQ(args=self.config)

    def train(self):
        # Init Memory
        expert_memory = Memory(self.config.env.expert_memory)
        policy_memory = Memory(self.config.env.policy_memory)
        expert_memory.load(self.data)

        steps = 0
        learn_step = 0
        begin_learn = False

        # Init trackers
        total_losses = []
        loss_1_list = []
        value_loss_list = []
        chi2_loss_list = []

        for _ in tqdm(range(self.config.train.epochs)):

            # For each epoch reset env
            state, _ = self.env.reset() # this will set random state
            done = False

            # For each episode
            for _ in range(self.config.train.episodes):

                # sample random action from the environment
                if steps < self.config.train.env_steps:
                    action = self.env.action_space.sample()
                
                # sample action from policy
                else:
                    action = self.agent.choose_action(state, sample=True)

                # take action in env
                next_state, _, done, _, _ = self.env.step(action)
                steps += 1
                policy_memory.add((state, next_state, action, 0.0, done))
                
                # Start training once memory is full
                if policy_memory.size() == self.config.env.policy_memory :
                    if not begin_learn:
                        print('Starting training loop!')
                        begin_learn = True
                    
                    # Train for learn steps
                    learn_step += 1
                    if learn_step == self.config.train.learn_steps:
                        print('Finished training loop!')
                        continue
                    
                    policy_batch = policy_memory.get_samples(
                        batch_size=self.config.train.batch_size,
                    )
                    expert_batch = expert_memory.get_samples(
                        batch_size = self.config.train.batch_size,
                    )
                    total_loss, loss_1, value_loss, chi2_loss = self.agent.iq_learn_update(
                        policy_batch=policy_batch,
                        expert_batch=expert_batch,
                    )
                
                if done:
                    break
                
                state = next_state

            # save tracked losses
            if begin_learn:
                total_losses.append(total_loss)
                loss_1_list.append(loss_1)
                value_loss_list.append(value_loss)
                chi2_loss_list.append(chi2_loss)
        
        # postprocess losses
        total_losses = [loss.item() for loss in total_losses]
        total_loss1 = [loss.item() for loss in loss_1_list]
        total_value_loss = [loss.item() for loss in value_loss_list]
        total_chi2_loss = [loss.item() for loss in chi2_loss_list]


        # plot losses over epochs
        length = len(total_losses)

        plt.figure(figsize=(10, 6))
        plt.plot(range(length), total_losses, label="Total Loss", color="black")
        plt.plot(range(length), total_loss1, label="IQ-Learn Loss", color="blue")
        plt.plot(range(length), total_value_loss, label="Value Loss", color="green")
        plt.plot(range(length), total_chi2_loss, label="Chi² Regularization Loss", color="red")

        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss Evolution During IQ-Learn Training")
        plt.legend()
        plt.grid()
        plt.show()

        return self.agent

            


        
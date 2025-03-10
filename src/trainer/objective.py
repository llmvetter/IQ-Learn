import logging

import gymnasium as gym
from ray import tune

from src.models.dotdict import DotDict
from src.environment.env import CarFollowingEnv
from src.dataset.preprocessor import MilanoPreprocessor
from src.dataset.expert_dataset import ExpertDataset
from src.trainer.trainer import Trainer
from src.trainer.evaluator import Evaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


gym.register(
    id="CarFollowing",
    entry_point=CarFollowingEnv,
)

def objective(config: DotDict):

    logging.info("Initializing Environment")
    env = gym.make(
        dataset_path=config.dataset_path,
        id = config.env.name,
        max_speed=config.env.max_v,
        max_distance=config.env.max_g,
        max_rel_speed=config.env.max_rel_v,
        actions=config.env.actions,
        delta_t=config.env.delta_t,
    )
    env = env.unwrapped
    env.reset()

    logging.info("Preprocessing Dataset")
    pp = MilanoPreprocessor(env)
    expert_trajectories = pp.preprocess(config.dataset_path)
    ex_data = ExpertDataset(expert_trajectories)

    logging.info("Initializing Trainer")
    trainer = Trainer(
        config=config,
        environment=env,
        expert_data=ex_data,
    )

    logging.info("Initializing training process")
    trained_agent = trainer.train()

    logging.info("Initializing Evaluator")
    evaluator = Evaluator(
        config=config,
        environment=env,
        agent=trained_agent,
    )
    logging.info("Initializing evaluation process")
    metrics = evaluator.evaluate(
        num_trajectories=100,
    )
    tune.report({'score': metrics['final_score']})

    return {'score': metrics['final_score']}
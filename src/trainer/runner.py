import logging

from ray import tune
from omegaconf import OmegaConf
import gymnasium as gym

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

def run_training(config):
    config = OmegaConf.load("/home/h6/leve469a/IQ-Learn/config.yaml")
    logging.info(f"loaded config with params: {config.__dict__}")

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
    trajs = pp.preprocess(path=config.dataset_path)
    ex_data = ExpertDataset(expert_trajectories=trajs)

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
    tune.report(final_score=metrics['final_score'])
    return metrics["final_score"]
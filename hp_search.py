
import sys

from ray import tune
from omegaconf import OmegaConf

from src.utils.utils import deep_merge
from src.trainer.objective import objective

# Init config
config = OmegaConf.load('/home/h6/leve469a/IQ-Learn/config.yaml')
config_dict = OmegaConf.to_container(config, resolve=True)

# Define hyperparameter search space
search_space = {
    'gamma': tune.uniform(0.95, 0.99),
    'method': {
        'alpha': tune.uniform(1, 5),
        'tanh': tune.choice([True, False]),
    },
    'agent': {
        'critic_lr': tune.loguniform(1e-6, 1e-4),
        'tau': tune.uniform(0.001, 0.05),
    }
}

task_id = int(sys.argv[1]) - 1

def create_name(trial):
    return f"trial_{trial.trial_id}"

# Init tuner
tuner = tune.Tuner(
    tune.with_parameters(objective),
    param_space=deep_merge(config_dict, search_space),
    tune_config=tune.TuneConfig(
        metric="score",
        mode="max",
        num_samples=2,
        trial_dirname_creator=create_name,
    ),
)

# Run tuning
results = tuner.fit()

# Get best config
best_result = results.get_best_result(metric='score', mode='max')
best_config = best_result.config if best_result else None

print(f'Best config found: {best_config}')

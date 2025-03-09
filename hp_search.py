from ray import tune
import sys
from omegaconf import OmegaConf
from src.trainer.runner import run_training

# Define hyperparameter search space
search_space = {
    "gamma": tune.uniform(0.95, 0.99),
    "method.alpha": tune.uniform(1, 5),
    "method.tanh": tune.choice([True, False]),
    "agent.critic_lr": tune.loguniform(1e-6, 1e-4),
    "agent.tau": tune.uniform(0.001, 0.05),
}

# SLURM Task ID for parallel runs
task_id = int(sys.argv[1]) - 1

# Init tuner
tuner = tune.Tuner(
    run_training,
    param_space=search_space,
    tune_config=tune.TuneConfig(
        metric="final_score",
        mode="max",
        num_samples=24,  # SLURM array size
    ),
)

# Run tuning
results = tuner.fit()

# Get best config
best_result = results.get_best_result(metric="final_score", mode="max")
best_config = best_result.config if best_result else None

print(f"Best config found: {best_config}")

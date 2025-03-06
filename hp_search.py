from ray import tune
import sys
from src.trainer.runner import run_training

# Define hyperparameter search space
search_space = {
    "gamma": tune.uniform(0.95, 0.99),
    "method": {
        "alpha": tune.uniform(1, 5),
        "tanh": tune.choice([True, False]),
    },
    "agent": {
        "critic_lr": tune.loguniform(1e-6, 1e-4),
        "tau": tune.uniform(0.001, 0.05),
    }
}

# SLURM Task ID for parallel runs
task_id = int(sys.argv[1]) - 1

# Run Ray Tune to generate configurations
analysis = tune.run(
    lambda config: run_training(config),
    config=search_space,
    num_samples=24,  # SLURM array size
    verbose=1
)

best_config = analysis.get_best_config(metric="final_score", mode="max")
print(f"Best config found: {best_config}")

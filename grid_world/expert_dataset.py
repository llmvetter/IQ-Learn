from typing import Any, Dict

import numpy as np
from torch.utils.data import Dataset


class ExpertDataset(Dataset):
    """Dataset for expert trajectories.

    Assumes expert dataset is a dict with keys {states, actions, rewards, lengths} with values containing a list of
    expert attributes of given shapes below. Each trajectory can be of different length.

    Expert rewards are not required but can be useful for evaluation.

        shapes:
            expert["states"]  =  [num_experts, traj_length, state_space]
            expert["actions"] =  [num_experts, traj_length, action_space]
            expert["rewards"] =  [num_experts, traj_length]
            expert["lengths"] =  [num_experts]
    """

    def __init__(self,
                 expert_trajectories: dict[str, list[Any]],
    ):
        """Subsamples an expert dataset from saved expert trajectories.

        Args:
            expert_location:          Location of saved expert trajectories.
        """
        self.trajectories = expert_trajectories
        self.i2traj_idx = {}
        self.length = np.array(self.trajectories["lengths"]).sum()
        traj_idx = 0
        i = 0

        # Convert flattened index i to trajectory indx and offset within trajectory
        self.get_idx = []

        for _j in range(self.length):
            while self.trajectories["lengths"][traj_idx] <= i:
                i -= self.trajectories["lengths"][traj_idx]
                traj_idx += 1

            self.get_idx.append((traj_idx, i))
            i += 1

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return self.length

    def __getitem__(self, i):
        traj_idx, i = self.get_idx[i]

        states = self.trajectories["states"][traj_idx][i]
        next_states = self.trajectories["next_states"][traj_idx][i]

        return (states,
                next_states,
                self.trajectories["actions"][traj_idx][i],
                self.trajectories["rewards"][traj_idx][i],
                self.trajectories["dones"][traj_idx][i])

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from environment.env import CarFollowingEnv
from typing import Any


class BasePreprocessor(ABC):
    def __init__(
            self,
            mdp: CarFollowingEnv,
    ):
        self.mdp = mdp
        self.min_speed = 5
        self.a_map = np.array(
            [v for v in self.mdp.action_mapping.values()]
        )
    @abstractmethod
    def preprocess(self, path: str) -> dict[str, list[str, Any]]:
        pass

class NapoliPreprocessor(BasePreprocessor):

    def _create_filtered_trajectory(
            self,
            df: pd.DataFrame,
            expert_num: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Creates filtered trajectory data for an expert and converts it to 
        the format required by ExpertDataset.

        Args:
            df: Filtered DataFrame of trajectories.
            expert_num: Index of the expert to process (e.g., 1, 2, or 3).
            min_speed: Minimum speed threshold for filtering.

        Returns:
            Tuple containing states, next_states, actions, rewards, dones, and trajectory length.
        """
        filtered_df = df[df[f'expert{expert_num}_speed'] > self.min_speed].copy()
        filtered_df = filtered_df.reset_index(drop=True)

        # States: Concatenate speed and distance
        states = np.stack([ 
            filtered_df[f'expert{expert_num}_speed'].values,
            filtered_df[f'expert{expert_num}_distance'].values
        ], axis=-1)

        # Actions: Expert acceleration
        actions = filtered_df[f'expert{expert_num}_acceleration'].values
        a_values = np.array(list(self.mdp.action_mapping.values()))
        a_keys = np.array(list(self.mdp.action_mapping.keys()))
        mapped_actions = a_keys[np.abs(actions[:, np.newaxis] - a_values).argmin(axis=1)]

        # Rewards: Placeholder (zero rewards for now)
        rewards = np.zeros_like(actions)

        # Compute next_states (next timestep in the trajectory)
        next_states = np.roll(states, shift=-1, axis=0)
        next_states[-1] = states[-1]  # For the last state, we keep the state as the next state (or use any placeholder)

        # Dones: Mark all timesteps except the last one as "not done" (0), and the last timestep as "done" (1)
        dones = np.zeros_like(actions)
        dones[-1] = 1  # The last state is done (episode end), might be better to take all velocity 0 as done and thereby chunk trajectories

        # Length of trajectory
        length = len(filtered_df)

        return states, next_states, mapped_actions, rewards, dones, length

    def preprocess(self, path: str) -> dict[str, list[str, Any]]:
        """
        Loads the dataset from the given path and processes it into the 
        required format for ExpertDataset.

        Args:
            path: Path to the trajectory file.

        Returns:
            Dictionary with keys "states", "next_states", "actions", "rewards", "dones", and "lengths".
        """
        # Load and preprocess the CSV file
        df = pd.read_csv(path, sep='\t', header=None)
        df['expert1_acceleration'] = df[0].diff().shift(-1)
        df['expert2_acceleration'] = df[1].diff().shift(-1)
        df['expert3_acceleration'] = df[2].diff().shift(-1)
        df['expert4_acceleration'] = df[3].diff().shift(-1)
        df = df.dropna()
        df = df.reset_index(drop=True)

        df.rename(columns={
            0: 'expert1_speed',
            1: 'expert2_speed',
            2: 'expert3_speed',
            3: 'expert4_speed',
            4: 'expert1_distance',
            5: 'expert2_distance',
            6: 'expert3_distance'
        }, inplace=True)

        # Process trajectories for each expert
        trajectories = []
        for expert_num in range(1, 4):  # Assuming 3 experts
            states, next_states, actions, rewards, dones, length = self._create_filtered_trajectory(
                df,
                expert_num,
            )
            trajectories.append({
                "states": states,
                "next_states": next_states,
                "actions": actions,
                "rewards": rewards,
                "dones": dones,
                "length": length
            })

        # Combine all trajectories into the ExpertDataset format
        expert_data = {
            "states": [traj["states"] for traj in trajectories],
            "next_states": [traj["next_states"] for traj in trajectories],
            "actions": [traj["actions"] for traj in trajectories],
            "rewards": [traj["rewards"] for traj in trajectories],
            "dones": [traj["dones"] for traj in trajectories],
            "lengths": [traj["length"] for traj in trajectories]
        }

        return expert_data

class MilanoPreprocessor(BasePreprocessor):

    def _filter_leader_follower_pairs(self, df, min_entries=1000):
        pair_counts = df.groupby(['Leader', 'Follower']).size()
        valid_pairs = pair_counts[pair_counts >= min_entries].index
        filtered_df = df[df.set_index(['Leader', 'Follower']).index.isin(valid_pairs)]
        return filtered_df

    def _create_filtered_trajectory(
            self,
            df: pd.DataFrame,
            leader: int,
            follower: int,
    ) -> dict[str, list[str, Any]]:
        subset = df[(df['Leader'] == leader) & (df['Follower'] == follower)]
        subset = subset.sort_values(by="Time [s]").reset_index(drop=True)
        states = np.array(list(zip(subset["Follower Speed"], subset["gap[m]"])))
        next_states = np.array(list(zip(subset["Follower Speed"].shift(-1), subset["gap[m]"].shift(-1))))

        states = states[:-1]
        next_states = next_states[:-1]
        
        actions = np.array(subset["Follower Tan. Acc."][:-1])
        a_values = np.array(list(self.mdp.action_mapping.values()))
        a_keys = np.array(list(self.mdp.action_mapping.keys()))
        mapped_actions = a_keys[np.abs(actions[:, np.newaxis] - a_values).argmin(axis=1)]

        rewards = np.zeros(len(actions))  # Maybe adjust for negative reward when too close
        dones = np.zeros(len(actions))  # Maybe adjust for done when too close
        length = len(actions)

        return {
            "states": states,
            "next_states": next_states,
            "actions": mapped_actions,
            "rewards": rewards,
            "dones": dones,
            "length": length
        }

    def preprocess(self, path: str) -> dict[str, list[str, Any]]:
        trajectories = []
        df_init = pd.read_csv(path)
        df_reduced = df_init[[
            'Time [s]',
            'Leader',
            'Follower',
            'Leader Speed',
            'Follower Speed',
            'Leader Tan. Acc.',
            'Follower Tan. Acc.',
            'Relative speed',
            'gap[m]',
        ]].copy()
        df = self._filter_leader_follower_pairs(df = df_reduced)
        unique_pairs = df.groupby(["Leader", "Follower"]).size().reset_index()

        for _, row in unique_pairs.iterrows():
            leader, follower = row["Leader"], row["Follower"]
            traj = self._create_filtered_trajectory(df, leader, follower)
            if traj["length"] > 0:
                trajectories.append(traj)
        expert_data = {
            "states": [traj["states"] for traj in trajectories],
            "next_states": [traj["next_states"] for traj in trajectories],
            "actions": [traj["actions"] for traj in trajectories],
            "rewards": [traj["rewards"] for traj in trajectories],
            "dones": [traj["dones"] for traj in trajectories],
            "lengths": [traj["length"] for traj in trajectories]
        }

        return expert_data

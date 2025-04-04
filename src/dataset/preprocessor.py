from abc import ABC, abstractmethod
from typing import Any
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

from src.environment.env import CarFollowingEnv


class BasePreprocessor(ABC):
    def __init__(
            self,
            mdp: CarFollowingEnv,
    ):
        self.mdp = mdp
        self.random_state = 42
        self.kmh_to_ms = 0.27778
        self.a_map = np.array(
            [v for v in self.mdp.action_mapping.values()]
        )
    @abstractmethod
    def preprocess(self, path: str) -> dict[str, list[str, Any]]:
        pass


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

        states = np.array(list(zip(
            subset["Follower Speed"],
            subset["gap[m]"],
            subset["Relative speed"],
        )))
        next_states = np.array(list(zip(
            subset["Follower Speed"].shift(-1),
            subset["gap[m]"].shift(-1),
            subset["Relative speed"].shift(-1),
        )))

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
        '''
        states will be returned in the format of [speed, gap, rel_speed]
        '''

        df_init = pd.read_csv(path)
        # df_init = df_init.iloc[::3].reset_index(drop=True)  # downsample to 10hz
        df_init["Follower Speed"] *= self.kmh_to_ms
        df_init["Relative speed"] *= self.kmh_to_ms
        
        df_reduced = df_init[[
            'Time [s]',
            'Leader',
            'Follower',
            'Follower Speed',
            'Leader Tan. Acc.',
            'Follower Tan. Acc.',
            'Relative speed',
            'gap[m]',
        ]].copy()

        df_filtered = self._filter_leader_follower_pairs(df_reduced)
        unique_pairs = df_filtered[['Leader', 'Follower']].drop_duplicates()

        train_pairs, test_pairs = train_test_split(
            unique_pairs,
            test_size=0.2,
            random_state=self.random_state,
        )

        train_df = df_filtered.merge(train_pairs, on=['Leader', 'Follower'])
        test_df = df_filtered.merge(test_pairs, on=['Leader', 'Follower'])

        trajectories = []
        for _, row in unique_pairs.iterrows():
            leader, follower = row["Leader"], row["Follower"]
            traj = self._create_filtered_trajectory(train_df, leader, follower)
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

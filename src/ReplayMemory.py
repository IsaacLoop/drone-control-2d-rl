from dataclasses import dataclass
import random
from collections import deque

import numpy as np
import torch


@dataclass
class Step:
    state: torch.Tensor
    action: torch.Tensor
    reward: float
    next_state: torch.Tensor
    done: bool
    h0_actor: torch.Tensor
    c0_actor: torch.Tensor


class LSTMReplayMemory:
    """
    Replay memory for storing experiences,
    typically to train a RL agent.

    Designed to work with an agent that uses LSTM,
    so it can store a sequence of experiences as well as the states
    h and c of the LSTM.
    """

    def __init__(self, n_episodes: int):
        """
        A ReplayMemory is defined by the number of episodes it can store,
        which is set by the n_episodes parameter. In this implementation,
        an episode can be infinite. It is expected that they are, in fact,
        finite.

        Args:
            n_episodes (int): The number of episodes the memory can store.
        """
        self.n_episodes = n_episodes
        self.episodes: dict[int, list[Step]] = {}
        self.episodes_ids: list[int] = []

    def push(
        self,
        episode_id: int,
        state: tuple[np.ndarray, np.ndarray],
        action: int,
        reward: float,
        next_state: tuple[np.ndarray, np.ndarray],
        h0_actor: np.ndarray,
        c0_actor: np.ndarray,
        done: bool,
    ):
        """Adds an experience to the memory.

        Args:
            episode_id (int): Unique identifier for the episode.
            state (np.ndarray): Drone state vector at time ``t``.
            action (int | np.ndarray): Action taken by the agent at time ``t``.
            reward (float): Reward received between ``t`` and ``t+1``.
            next_state (np.ndarray): Drone state vector at time ``t+1``.
            done (bool): Whether state ``t+1`` is terminal.
            h0_actor (np.ndarray): Initial hidden state of the actor's LSTM.
            c0_actor (np.ndarray): Initial cell state of the actor's LSTM.
        """

        if episode_id not in self.episodes:
            self.episodes[episode_id] = []
            self.episodes_ids.append(episode_id)
            if len(self.episodes_ids) > self.n_episodes:
                self.episodes.pop(self.episodes_ids[0])
                self.episodes_ids.pop(0)

        self.episodes[episode_id].append(
            Step(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                h0_actor=h0_actor,
                c0_actor=c0_actor,
            )
        )

    def sample(self, batch_size: int, seq_len: int = 32) -> list[Step]:
        """
        Samples a batch of experiences from the memory.

        Args:
            batch_size (int): The number of episodes to sample a slice from.
            seq_len (int, optional):
                The length of the slice to sample from each selected episode.
                Defaults to 32.

        Raises:
            ValueError: If there are not enough episodes with at least `seq_len` steps.

        Returns:
            list[Step]: A list of sampled experiences.
        """
        long_enough_episodes = [e for e in self.episodes.values() if len(e) >= seq_len]
        if len(long_enough_episodes) < batch_size:
            raise ValueError(
                f"Not enough episodes with at least {seq_len} steps, got {len(long_enough_episodes)}"
            )

        batch = []
        for e in random.sample(long_enough_episodes, batch_size):
            start = random.randint(0, len(e) - seq_len)
            batch.extend(e[start : start + seq_len])
        return batch

    def __len__(self):
        """Returns the number of episodes in the memory.

        Returns:
            int: The number of episodes in the memory.
        """
        return len(self.episodes)

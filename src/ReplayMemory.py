import random
from collections import deque

class ReplayMemory:
    """
    Replay memory for storing experiences,
    typically to train a RL agent.
    """

    def __init__(self, capacity: int):
        """
        A ReplayMemory is defined by the number of experiences it can store,
        which is set by the capacity parameter.

        Args:
            capacity (int): The maximum number of experiences the memory can store.
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Adds an experience to the memory.

        Args:
            state (tuple[np.ndarray, np.ndarray]): A state at time t. A grid and a fruit type.
            action (int): The action taken by the agent when looking at state t.
            reward (float): The reward received by the agent between state t and t+1.
            next_state (tuple[np.ndarray, np.ndarray]): The next state of the environment, at time t+1. A grid and a fruit type.
            done (bool): Whether state t+1 is a terminal state (game over).
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """Samples a batch of experiences from the memory.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            list[tuple]: A list of experiences.
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        """Returns the number of experiences in the memory.

        Returns:
            int: The number of experiences in the memory.
        """
        return len(self.buffer)
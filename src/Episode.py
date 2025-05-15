import math
import random
from abc import ABC, abstractmethod

import numpy as np

from src.Game import Game

STRAIGHT_LINE_CHANGE_DIRECTION_FRAMES_INTERVAL = 90


def build_state(env: Game, desired_vx: float, desired_vy: float) -> np.ndarray:
    vx, vy = env.env.drone_velocity / 5
    va = env.env.ang_vel / 10
    a_cos = math.cos(env.drone_angle)
    a_sin = math.sin(env.drone_angle)
    propL = env.env.drone.L_speed
    propR = env.env.drone.R_speed
    return np.array(
        [
            vx,
            vy,
            va,
            a_cos,
            a_sin,
            propL,
            propR,
            desired_vx / 5,
            desired_vy / 5,
        ],
        dtype=np.float32,
    )


def speed_error_reward(error: float) -> float:
    """
    Linear minus 5, grows faster near 0, with a max of 0.
    Error should generally be positive (absolute).
    Reward is always negative.
    """
    return 5 * np.exp(-4.78 * abs(error)) - abs(error) - 5


class AbstractEpisode(ABC):

    def __init__(self, duration_steps: int, dt: float, gui: bool):
        self.duration_steps = duration_steps
        self.dt = dt
        self.t = 0

        self.game = Game(gui=gui, human_player=False, dt=dt, wind=True, rain=True)
        self.desired_velocity = np.array([0.0, 0.0])
        self.done = False
        self._configure_environment()

    @abstractmethod
    def _configure_environment(self):
        """Initialise `self.game` to suit the task."""
        ...

    @abstractmethod
    def _compute_reward(self) -> float:
        """Compute per-step reward using `self.game` state."""
        ...

    #  Public API
    @property
    def state(self) -> np.ndarray:
        return build_state(
            self.game, self.desired_velocity[0], self.desired_velocity[1]
        )

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool]:
        aL, aR = float(action[0]), float(action[1])
        self.game.step(aL, aR)
        if self.game.gui:
            self.game.render()
        reward = self._compute_reward()
        self.t += 1
        self.done = self.t >= self.duration_steps
        return self.state, reward, self.done


class StraightLineEpisode(AbstractEpisode):
    """The drone must fly in a straight line at a certain speed for some time."""

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool]:
        r = super().step(action)
        if random.random() < 1 / STRAIGHT_LINE_CHANGE_DIRECTION_FRAMES_INTERVAL:
            self.desired_velocity = np.random.uniform(-5, 5, size=2)
        return r

    def _configure_environment(self):
        # Drone init
        self.game.set_drone_velocity(*np.random.uniform(-5, 5, size=2))
        self.game.set_drone_angle(np.random.uniform(0, 2 * math.pi))
        self.game.set_drone_propeller_speeds(*np.random.uniform(-1, 1, size=2))

        # Desired
        self.desired_velocity = np.random.uniform(-5, 5, size=2)

    def _compute_reward(self) -> float:
        velocity_error = np.linalg.norm(
            self.game.drone_velocity - self.desired_velocity
        )
        return speed_error_reward(velocity_error)


class StopEpisode(AbstractEpisode):
    """Drone is moving in a certain direction with a certain angle,
    and it must stop as quickly as possible, and face upwards."""

    def _configure_environment(self):
        # Drone init
        self.game.set_drone_velocity(*np.random.uniform(-5, 5, size=2))
        self.game.set_drone_angle(np.random.uniform(0, 2 * math.pi))
        self.game.set_drone_propeller_speeds(*np.random.uniform(-1, 1, size=2))

        # Desired
        self.desired_velocity = np.array([0.0, 0.0])

    def _compute_reward(self) -> float:
        velocity_error = np.linalg.norm(self.game.drone_velocity)
        return speed_error_reward(velocity_error)

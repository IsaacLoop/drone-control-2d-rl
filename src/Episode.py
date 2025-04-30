import math
from abc import ABC, abstractmethod

import numpy as np

from src.Game import Game


def build_state(env: Game, desired_vx: float, desired_vy: float) -> np.ndarray:
    vx, vy = env.env.drone_velocity / 5
    va = env.env.ang_vel / 10
    a_cos = math.cos(env.env.drone_angle)
    a_sin = math.sin(env.env.drone_angle)
    wind_vx = env.env.wind_force / 10
    rain_vy = env.env.rain_force / 10
    propL = env.env.drone.L_speed
    propR = env.env.drone.R_speed
    return np.array(
        [
            vx,
            vy,
            va,
            a_cos,
            a_sin,
            wind_vx,
            rain_vy,
            propL,
            propR,
            desired_vx / 5,
            desired_vy / 5,
        ],
        dtype=np.float32,
    )


class AbstractEpisode(ABC):

    def __init__(self, duration_steps: int, dt: float, gui: bool):
        self.duration_steps = duration_steps
        self.dt = dt
        self.t = 0
        self.game = Game(gui=gui, human_player=False, dt=dt)
        self.desired_velocity = np.array([0.0, 0.0])
        self.done = False
        self._configure_environment()

    @abstractmethod
    def _configure_environment(self):
        """Initialise `self.env` to suit the task."""
        ...

    @abstractmethod
    def _compute_reward(self) -> float:
        """Compute per-step reward using `self.env` state."""
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

    def _configure_environment(self):
        # Drone init
        self.game.set_drone_velocity(*np.random.uniform(-5, 5, size=2))
        self.game.set_drone_angle(np.random.uniform(0, 2 * math.pi))
        self.game.set_drone_propeller_speeds(*np.random.uniform(-1, 1, size=2))

        # Desired
        self.desired_velocity = np.random.uniform(-5, 5, size=2)

    def _compute_reward(self) -> float:
        v_error = self.game.drone_velocity - self.desired_velocity
        speed = np.linalg.norm(v_error)

        v_parallel = np.dot(v_error, self.desired_velocity) / (speed + 1e-6)
        v_perpendicular = np.sqrt(np.linalg.norm(v_error) ** 2 - v_parallel**2)

        return -1 * abs(v_parallel) - 2 * abs(v_perpendicular)


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
        v_error = self.game.drone_velocity
        speed = np.linalg.norm(v_error)

        angle_normalized = math.atan2(
            math.sin(self.game.drone_angle), math.cos(self.game.drone_angle)
        )
        angle_error = abs(angle_normalized)

        return -2 * abs(speed) - 1 * abs(angle_error)

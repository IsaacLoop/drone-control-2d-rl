import numpy as np


class OUNoise:
    """
    Ornstein-Uhlenbeck noise.
    Used for wind and rain.
    """

    def __init__(
        self, size: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2
    ):
        self.mu = float(mu)
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.size = int(size)
        self.state = np.ones(self.size, dtype=np.float64) * self.mu

    def reset(self):
        self.state.fill(self.mu)

    def sample(self) -> np.ndarray:
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(
            self.size
        )
        self.state = self.state + dx
        return self.state


class Drone:
    """
    Drone class. Only used to manage drone geometry and motor dynamics.
    I am actually not sure if a whole standalone class is necessary here,
    but hey, it works.
    """

    def __init__(self):
        self.x_length = 0.30
        self.y_length = 0.05
        self.L_xy = (-self.x_length / 2.0, 0.0)  # left propeller position
        self.R_xy = (self.x_length / 2.0, 0.0)  # right propeller position

        # Just enough to counter gravity exactly when spawned
        self.L_speed = 0.4905
        self.R_speed = 0.4905

        # Motor parameters
        self.max_prop_acceleration = 4.0

    def accelerate_prop(self, aL: float, aR: float, dt: float) -> None:
        """Update propeller speeds with a saturating first-order response.

        *aL* and *aR* are command signals in [-1, 1] (negative = reverse).
        The closer a propeller is to its limit (|speed| -> 1), the slower it
        accelerates. When the command is zero, the prop decays exponentially.
        """
        half_life = 1.0  # seconds
        decay_factor = 0.5 ** (dt / half_life)

        def _update(propeller_speed: float, cmd: float) -> float:
            if abs(cmd) < 1e-6:
                return propeller_speed * decay_factor

            cmd_clipped = np.clip(cmd, -1.0, 1.0)

            if cmd_clipped > 0:
                effective_accel_factor = cmd_clipped * (1.0 - propeller_speed)
            elif cmd_clipped < 0:
                effective_accel_factor = cmd_clipped
            else:
                effective_accel_factor = 0.0

            acceleration = effective_accel_factor * self.max_prop_acceleration
            new_speed = propeller_speed + acceleration * dt
            return np.clip(new_speed, 0.0, 1.0)

        self.L_speed = _update(self.L_speed, aL)
        self.R_speed = _update(self.R_speed, aR)


class Environment:
    """
    Simple 2D physics world holding a drone, and having wind and rain
    that impact the drone velocity.
    """

    def __init__(
        self,
        wind_theta: float = 0.00002,
        wind_sigma: float = 0.004,
        rain_theta: float = 0.00002,
        rain_sigma: float = 0.004,
    ):
        self.drone = Drone()

        self.gravity = -9.81
        self.mass = 1.0
        self.max_thrust = 10.0  # N per propeller at full power

        # Aerodynamic constants that follow loosely
        # the drag formula: F_drag = 1/2 * rho * Cd * v^2 * A
        rho = 1.225
        Cd = 1.05
        area = self.drone.x_length * self.drone.y_length
        self.linear_drag_k = 0.5 * rho * Cd * area
        self.rotational_drag_k = 0.02

        # Drone state
        self.pos = np.array([0.0, self.drone.y_length / 2.0])  # m
        self.vel = np.zeros(2)  # m s^-1
        self.angle = 0.0  # rad (0 = level)
        self.ang_vel = 0.0  # rad s^-1

        w, h = self.drone.x_length, self.drone.y_length
        self.inertia = (1.0 / 12.0) * self.mass * (w * w + h * h)

        self.wind_noise = OUNoise(size=1, mu=0.0, theta=wind_theta, sigma=wind_sigma)
        self.rain_noise = OUNoise(size=1, mu=0.0, theta=rain_theta, sigma=rain_sigma)
        for _ in range(1000):
            self.wind_noise.sample()
            self.rain_noise.sample()

        self.wind_vx = 0.0
        self.rain_vy = 0.0

    def step(self, aL: float, aR: float, dt: float, substeps: int = 10) -> None:
        """
        Advance the world by *dt* seconds.
        """
        sub_dt = dt / max(1, substeps)
        for _ in range(substeps):
            self.drone.accelerate_prop(aL, aR, sub_dt)
            self.wind_vx = float(self.wind_noise.sample()[0])
            self.rain_vy = min(0.0, float(self.rain_noise.sample()[0]))
            self._physics_step(sub_dt)

    def _physics_step(self, dt: float) -> None:
        """Compute forces & torques then integrate state forward by *dt*."""

        sin_a, cos_a = np.sin(self.angle), np.cos(self.angle)
        drone_x_axis = np.array([cos_a, sin_a])
        drone_up = np.array([-sin_a, cos_a])

        Fg = np.array([0.0, self.mass * self.gravity])

        thrusts = []
        torques = []
        for speed, (prop_x, _) in (
            (self.drone.L_speed, self.drone.L_xy),
            (self.drone.R_speed, self.drone.R_xy),
        ):
            thrust_mag = speed * self.max_thrust
            F_t = drone_up * thrust_mag
            thrusts.append(F_t)

            # Torque = r × F
            r_body = np.array([prop_x, 0.0])
            r_world = drone_x_axis * r_body[0]
            torque_z = np.cross(r_world, F_t)
            torques.append(torque_z)

        # Air drag (linear)
        v_air = np.array([self.wind_vx, self.rain_vy])
        rel_v = self.vel - v_air
        F_drag = -self.linear_drag_k * rel_v * np.linalg.norm(rel_v)

        # Air drag (angular)
        T_drag = -self.rotational_drag_k * self.ang_vel * abs(self.ang_vel)

        F_total = Fg + F_drag + thrusts[0] + thrusts[1]
        T_total = torques[0] + torques[1] + T_drag

        acc = F_total / self.mass
        self.vel += acc * dt
        self.pos += self.vel * dt

        ang_acc = T_total / self.inertia
        self.ang_vel += ang_acc * dt
        self.angle += self.ang_vel * dt

        self.angle = (self.angle + np.pi) % (2 * np.pi) - np.pi

    # Getters and setters
    @property
    def wind_force(self) -> float:
        """Wind force in N, on the horizontal axis from left (negative) to right (positive)."""
        return self.wind_vx

    @property
    def rain_force(self) -> float:
        """Rain force in N, on the vertical axis. Negative only (downwards)."""
        return self.rain_vy

    @property
    def drone_position(self) -> np.ndarray:
        """Drone position, relative to the world origin."""
        return self.pos.copy()

    @property
    def drone_angle(self) -> float:
        """Drone angle in radians. An angle of 0 means the drone is facing up, correctly leveled."""
        return self.angle

    @property
    def drone_velocity(self) -> np.ndarray:
        """Drone velocity on x and y axes."""
        return self.vel.copy()

    def set_drone_position(self, x: float, y: float) -> None:
        """Set drone position, relative to the world origin."""
        self.pos = np.array([x, y], dtype=float)

    def set_drone_angle(self, angle: float) -> None:
        """Angle is in radians. An angle of 0 means the drone is facing up, correctly leveled."""
        self.angle = (angle + np.pi) % (2 * np.pi) - np.pi

    def set_drone_velocity(self, vx: float, vy: float) -> None:
        """Set drone velocity on x and y axes."""
        self.vel = np.array([vx, vy], dtype=float)

    def set_drone_angular_velocity(self, va: float) -> None:
        """Set drone angular velocity in radians per second."""
        self.ang_vel = float(va)

    def set_drone_propeller_speeds(self, L_speed: float, R_speed: float) -> None:
        """Set drone propeller speeds."""
        self.drone.L_speed = L_speed
        self.drone.R_speed = R_speed

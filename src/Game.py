import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import numpy as np
import pygame

from src.Environment import Environment


class Game:
    """
    A Game contains an Environment, and gives controls over the environment.

    Those controls can be either directly used by a human player, with a keyboard,
    or used with the API of the Game object.
    """

    def __init__(
        self,
        gui: bool,
        human_player: bool,
        dt: float,
        wind: bool = True,
        rain: bool = True,
    ):
        assert not (
            not gui and human_player
        ), "Cannot have a human player without a GUI"
        self.gui = gui
        self.human_player = human_player
        self.env = Environment(
            wind_theta=0.00002 if wind else 0.0,
            wind_sigma=0.004 if wind else 0.0,
            rain_theta=0.00002 if rain else 0.0,
            rain_sigma=0.004 if rain else 0.0,
        )
        self.dt = dt
        self.is_running = True

        self.aL = 0.0
        self.aR = 0.0

        if self.gui:
            pygame.init()

            self.window_width = 1000
            self.window_height = 1000
            self.screen = pygame.display.set_mode(
                (self.window_width, self.window_height)
            )
            pygame.display.set_caption("Drone")

            self.font = pygame.font.SysFont(None, int(24))
            self.clock = pygame.time.Clock()
            self.pixels_per_metre: float = 100.0  # Pixels per metre for rendering scale

    def step(self, aL: float, aR: float):
        if not self.is_running:
            print("Game done.")
            return
        self.aL, self.aR = aL, aR
        self.env.step(self.aL, self.aR, self.dt)

    def handle_events(self, control_type: str):
        """
        control_type: "direct" or "arrow"

        direct -> direct control of thrust
        arrow -> control of direction with arrow keys
        """
        if not self.gui:
            return None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.is_running = False
                return None

        if not self.human_player:
            return None

        keys = pygame.key.get_pressed()
        if control_type == "direct":
            return {
                "A": keys[pygame.K_a],
                "Q": keys[pygame.K_q],
                "P": keys[pygame.K_p],
                "M": keys[pygame.K_m],
            }
        elif control_type == "arrow":
            return {
                "UP": keys[pygame.K_UP],
                "DOWN": keys[pygame.K_DOWN],
                "LEFT": keys[pygame.K_LEFT],
                "RIGHT": keys[pygame.K_RIGHT],
            }
        return None

    def render(self):
        """
        To be completely honest, this whole method was written by ChatGPT.
        It's in no way a critical part of the project, it doesn't really matter,
        and I am not brave enough to handle GUI engineering if I don't have to.
        """
        if not self.gui:
            return

        self.screen.fill((0, 0, 0))
        ppm = self.pixels_per_metre  # shorthand

        # Get screen dimensions and center (in pixels)
        screen_width, screen_height = self.screen.get_size()
        center_x, center_y = screen_width // 2, screen_height // 2

        # Drone's world position (metres) - this is our camera focus
        # drone_pos_m = self.env.body.position
        drone_pos_m = self.env.drone_position

        # --- Draw Grid --- #
        # Define grid spacing in world units (metres)
        grid_spacing_m = 0.5  # e.g., lines every 0.5 metres
        grid_spacing_px = grid_spacing_m * ppm
        grid_color = (50, 50, 50)  # Dark Gray

        # Calculate the world coordinates (metres) of the top-left corner relative to the drone
        # Drone is at screen center (center_x, center_y)
        # Top-left screen pixel corresponds to world offset (-center_x / ppm, +center_y / ppm) from drone
        world_offset_left_m = -center_x / ppm
        world_offset_top_m = center_y / ppm  # Pygame Y is inverted

        # World coordinates (metres) of the top-left screen corner
        world_view_left_m = drone_pos_m[0] + world_offset_left_m
        world_view_top_m = drone_pos_m[1] + world_offset_top_m

        # Calculate the first vertical grid line coordinate >= world_view_left_m
        start_x_m = np.floor(world_view_left_m / grid_spacing_m) * grid_spacing_m
        # Calculate the first horizontal grid line coordinate <= world_view_top_m
        start_y_m = np.ceil(world_view_top_m / grid_spacing_m) * grid_spacing_m

        # Calculate visible world width and height in metres
        world_view_width_m = screen_width / ppm
        world_view_height_m = screen_height / ppm

        # Draw vertical lines
        num_vert_lines = int(np.ceil(world_view_width_m / grid_spacing_m)) + 2
        for i in range(num_vert_lines):
            x_m = start_x_m + i * grid_spacing_m
            # Convert world x (metres) to screen x (pixels)
            screen_x = center_x + (x_m - drone_pos_m[0]) * ppm
            pygame.draw.line(
                self.screen, grid_color, (screen_x, 0), (screen_x, screen_height)
            )

        # Draw horizontal lines
        num_horz_lines = int(np.ceil(world_view_height_m / grid_spacing_m)) + 2
        for i in range(num_horz_lines):
            y_m = (
                start_y_m - i * grid_spacing_m
            )  # Subtract because Y decreases downwards in world
            # Convert world y (metres) to screen y (pixels)
            screen_y = center_y - (y_m - drone_pos_m[1]) * ppm  # Invert Y
            pygame.draw.line(
                self.screen, grid_color, (0, screen_y), (screen_width, screen_y)
            )

        # --- Draw Drone ---
        # pos_m = self.env.body.position # metres
        pos_m = self.env.drone_position  # metres
        angle = self.env.drone_angle  # radians

        # Calculate drone vertices in local coordinates (metres) relative to body center
        half_width_m = self.env.drone.x_length / 2.0
        half_height_m = self.env.drone.y_length / 2.0
        local_points_m = [
            (-half_width_m, -half_height_m),
            (half_width_m, -half_height_m),
            (half_width_m, half_height_m),
            (-half_width_m, half_height_m),
        ]

        # Rotate and transform local points (metres) to screen coordinates (pixels)
        screen_points = []
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        for lx_m, ly_m in local_points_m:
            # Rotate local point
            rotated_local_x_m = lx_m * cos_a - ly_m * sin_a
            rotated_local_y_m = lx_m * sin_a + ly_m * cos_a

            # Calculate screen coordinates relative to the screen center
            # The drone's center (pos_m) is always at the screen center (center_x, center_y)
            screen_rel_x_px = rotated_local_x_m * ppm
            screen_rel_y_px = rotated_local_y_m * ppm  # Y is not inverted yet

            # Final screen coordinates
            screen_x = center_x + screen_rel_x_px
            screen_y = center_y - screen_rel_y_px  # Invert Y-axis here

            # Clamp coordinates (optional, but good practice)
            buffer = max(screen_width, screen_height)  # Smaller buffer might suffice
            min_coord = -buffer
            max_x_coord = screen_width + buffer
            max_y_coord = screen_height + buffer
            clamped_x = max(min_coord, min(int(round(screen_x)), max_x_coord))
            clamped_y = max(min_coord, min(int(round(screen_y)), max_y_coord))

            screen_points.append((clamped_x, clamped_y))

        # Draw drone polygon
        pygame.draw.polygon(self.screen, (255, 255, 255), screen_points)  # White

        # --- Draw Orientation Marker (^) ---
        # Define marker size and offsets in world units (metres)
        marker_height_m = 0.08  # 8 cm tall
        marker_width_m = 0.04  # 4 cm wide base
        marker_base_y_offset_m = half_height_m + 0.02  # 2 cm above drone top surface

        # Define the marker points in local coordinates (metres) relative to the drone's center (0,0)
        local_p_tip_m = (0, marker_base_y_offset_m + marker_height_m)
        local_p_base_left_m = (-marker_width_m / 2, marker_base_y_offset_m)
        local_p_base_right_m = (marker_width_m / 2, marker_base_y_offset_m)

        marker_points_local_m = [
            local_p_base_left_m,
            local_p_tip_m,
            local_p_base_right_m,
        ]
        marker_points_screen = []

        # Rotate and transform marker points to screen coordinates
        for lx_m, ly_m in marker_points_local_m:
            # Rotate local point
            rotated_local_x_m = lx_m * cos_a - ly_m * sin_a
            rotated_local_y_m = lx_m * sin_a + ly_m * cos_a

            # Transform to screen coordinates (relative to center, scaled, Y inverted)
            screen_rel_x_px = rotated_local_x_m * ppm
            screen_rel_y_px = rotated_local_y_m * ppm
            screen_x = center_x + screen_rel_x_px
            screen_y = center_y - screen_rel_y_px  # Invert Y

            marker_points_screen.append((int(round(screen_x)), int(round(screen_y))))

        # Draw the marker lines (e.g., in Red)
        marker_color = (255, 0, 0)  # Red
        pygame.draw.line(
            self.screen,
            marker_color,
            marker_points_screen[0],
            marker_points_screen[1],
            2,
        )  # Line base_left -> tip
        pygame.draw.line(
            self.screen,
            marker_color,
            marker_points_screen[1],
            marker_points_screen[2],
            2,
        )  # Line tip -> base_right

        # --- Draw Thrust Indicators ---
        max_thrust_line_length_m = 0.3  # Visual length in metres for max thrust
        thrust_color_positive = (0, 150, 255)  # Light blue
        thrust_color_negative = (255, 100, 0)  # Orange/Red

        prop_data = [
            (self.env.drone.L_speed, self.env.drone.L_xy),
            (self.env.drone.R_speed, self.env.drone.R_xy),
        ]

        for speed, local_prop_pos_m in prop_data:
            if abs(speed) < 0.01:  # Don't draw tiny lines
                continue

            thrust_ratio = speed  # speed is already -1 to 1
            line_length_m = thrust_ratio * max_thrust_line_length_m

            # Calculate start point (propeller location)
            lx_m, ly_m = local_prop_pos_m
            # Rotate
            rotated_start_local_x_m = lx_m * cos_a - ly_m * sin_a
            rotated_start_local_y_m = lx_m * sin_a + ly_m * cos_a
            # Transform to screen
            start_screen_rel_x_px = rotated_start_local_x_m * ppm
            start_screen_rel_y_px = rotated_start_local_y_m * ppm
            start_screen_x = center_x + start_screen_rel_x_px
            start_screen_y = center_y - start_screen_rel_y_px  # Invert Y
            start_point_screen = (
                int(round(start_screen_x)),
                int(round(start_screen_y)),
            )

            # Calculate end point (offset from start point along drone's local Y axis)
            # End point relative to propeller in local drone frame
            end_offset_local_m = (0, line_length_m)
            # End point relative to drone center in local drone frame
            end_point_local_m = (
                lx_m + end_offset_local_m[0],
                ly_m + end_offset_local_m[1],
            )
            # Rotate
            rotated_end_local_x_m = (
                end_point_local_m[0] * cos_a - end_point_local_m[1] * sin_a
            )
            rotated_end_local_y_m = (
                end_point_local_m[0] * sin_a + end_point_local_m[1] * cos_a
            )
            # Transform to screen
            end_screen_rel_x_px = rotated_end_local_x_m * ppm
            end_screen_rel_y_px = rotated_end_local_y_m * ppm
            end_screen_x = center_x + end_screen_rel_x_px
            end_screen_y = center_y - end_screen_rel_y_px  # Invert Y
            end_point_screen = (int(round(end_screen_x)), int(round(end_screen_y)))

            # Choose color
            color = thrust_color_positive if thrust_ratio > 0 else thrust_color_negative

            # Draw line
            pygame.draw.line(
                self.screen, color, start_point_screen, end_point_screen, 3
            )

        # --- Display Text ---
        # Text display uses world coordinates/velocities directly from env, which are already in metres/m/s
        text_y = 10  # Starting Y position for text

        # Drone Coordinates (metres)
        coord_text = f"Pos (m): ({self.env.drone_position[0]:.2f}, {self.env.drone_position[1]:.2f})"
        coord_surface = self.font.render(
            coord_text, True, (255, 255, 255)
        )  # White text
        self.screen.blit(coord_surface, (10, text_y))
        text_y += coord_surface.get_height() + 2  # Add small padding

        # Angle
        angle_text = (
            f"Angle: {np.degrees(self.env.drone_angle):.2f}°"  # Display in degrees
        )
        angle_surface = self.font.render(angle_text, True, (255, 255, 255))
        self.screen.blit(angle_surface, (10, text_y))
        text_y += angle_surface.get_height() + 2

        # Linear Velocity
        vx_text = f"Vx: {self.env.drone_velocity[0]:.2f}"
        vx_surface = self.font.render(vx_text, True, (255, 255, 255))
        self.screen.blit(vx_surface, (10, text_y))
        text_y += vx_surface.get_height() + 2

        vy_text = f"Vy: {self.env.drone_velocity[1]:.2f}"
        vy_surface = self.font.render(vy_text, True, (255, 255, 255))
        self.screen.blit(vy_surface, (10, text_y))
        text_y += vy_surface.get_height() + 2

        # Angular Velocity
        va_text = f"Va: {self.env.ang_vel:.2f}"
        va_surface = self.font.render(va_text, True, (255, 255, 255))
        self.screen.blit(va_surface, (10, text_y))
        text_y += va_surface.get_height() + 2

        # Wind Speed
        # Need to access wind_vx from environment, assuming it's stored after step
        try:
            wind_vx = self.env.wind_vx
        except AttributeError:
            wind_vx = 0  # Default if not set yet
        wind_text = f"Wind: {wind_vx:.2f} m/s"
        wind_surface = self.font.render(wind_text, True, (255, 255, 255))  # White text
        self.screen.blit(wind_surface, (10, text_y))
        text_y += wind_surface.get_height() + 2  # Add small padding

        # Rain Speed
        # Need to access rain_vy from environment, assuming it's stored after step
        try:
            rain_vy = self.env.rain_vy
        except AttributeError:
            rain_vy = 0  # Default if not set yet
        rain_text = f"Rain: {rain_vy:.2f} m/s"
        rain_surface = self.font.render(rain_text, True, (255, 255, 255))  # White text
        self.screen.blit(rain_surface, (10, text_y))
        text_y += rain_surface.get_height() + 2  # Add small padding

        # Propeller Power
        prop_text = f"Prop L: {self.env.drone.L_speed:.2f}, Prop R: {self.env.drone.R_speed:.2f}"
        prop_surface = self.font.render(prop_text, True, (255, 255, 255))  # White text
        self.screen.blit(prop_surface, (10, text_y))

        # --- Draw Wind/Rain Vectors (Top Right) ---
        vector_origin_x = self.window_width - 100
        vector_origin_y = 100
        vector_scale = 50.0  # Pixels per m/s
        vector_origin_screen = (vector_origin_x, vector_origin_y)

        # Get wind/rain data
        wind_vx = self.env.wind_vx
        rain_vy = (
            self.env.rain_vy
        )  # Remember: positive rain_vy in physics means downwards

        # Calculate vector end points in screen coordinates
        # Wind vector (Red) - horizontal
        wind_end_x = vector_origin_x + wind_vx * vector_scale
        wind_end_y = vector_origin_y  # No vertical component
        wind_end_screen = (int(round(wind_end_x)), int(round(wind_end_y)))

        # Rain vector (Blue) - vertical
        rain_end_x = vector_origin_x  # No horizontal component
        rain_end_y = (
            vector_origin_y - rain_vy * vector_scale
        )  # Positive physics rain_vy decreases screen Y (upwards)
        rain_end_screen = (int(round(rain_end_x)), int(round(rain_end_y)))

        # Resultant vector (White)
        resultant_end_x = vector_origin_x + wind_vx * vector_scale
        resultant_end_y = (
            vector_origin_y - rain_vy * vector_scale
        )  # Apply same fix for resultant
        resultant_end_screen = (
            int(round(resultant_end_x)),
            int(round(resultant_end_y)),
        )

        # Draw vectors
        pygame.draw.line(
            self.screen, (255, 0, 0), vector_origin_screen, wind_end_screen, 2
        )  # Red Wind
        pygame.draw.line(
            self.screen, (0, 0, 255), vector_origin_screen, rain_end_screen, 2
        )  # Blue Rain
        pygame.draw.line(
            self.screen, (255, 255, 255), vector_origin_screen, resultant_end_screen, 2
        )  # White Resultant

        # Optional: Draw a small circle at the origin
        pygame.draw.circle(self.screen, (200, 200, 200), vector_origin_screen, 3)

        pygame.display.flip()
        self.clock.tick(1 / self.dt)

    @property
    def wind_force(self) -> float:
        """Wind force in N, on the horizontal axis from left (negative) to right (positive)."""
        return self.env.wind_force

    @property
    def rain_force(self) -> float:
        """Rain force in N, on the vertical axis. Negative only (downwards)."""
        return self.env.rain_force

    @property
    def drone_position(self) -> np.ndarray:
        """Drone position, relative to the world origin."""
        return self.env.drone_position

    @property
    def drone_angle(self) -> float:
        """Drone angle in radians. An angle of 0 means the drone is facing up, correctly leveled."""
        return self.env.drone_angle

    @property
    def drone_velocity(self) -> np.ndarray:
        """Drone velocity on x and y axes."""
        return self.env.drone_velocity

    def set_drone_position(self, x: float, y: float) -> None:
        """Set drone position, relative to the world origin."""
        self.env.set_drone_position(x, y)

    def set_drone_angle(self, angle: float) -> None:
        """Angle is in radians. An angle of 0 means the drone is facing up, correctly leveled."""
        self.env.set_drone_angle(angle)

    def set_drone_velocity(self, vx: float, vy: float) -> None:
        """Set drone velocity on x and y axes."""
        self.env.set_drone_velocity(vx, vy)

    def set_drone_angular_velocity(self, va: float) -> None:
        """Set drone angular velocity in radians per second."""
        self.env.set_drone_angular_velocity(va)

    def set_drone_propeller_speeds(self, L_speed: float, R_speed: float) -> None:
        """Set drone propeller speeds."""
        self.env.set_drone_propeller_speeds(L_speed, R_speed)


if __name__ == "__main__":
    fps = 30
    game = Game(gui=True, human_player=True, dt=1 / fps)

    while game.is_running:
        keys = game.handle_events(control_type="direct")
        if keys is None:
            continue

        aL, aR = 0.0, 0.0
        if keys["A"] and keys["Q"]:
            aL = 0.0
        elif keys["A"]:
            aL = 1.0
        elif keys["Q"]:
            aL = -1.0

        if keys["P"] and keys["M"]:
            aR = 0.0
        elif keys["P"]:
            aR = 1.0
        elif keys["M"]:
            aR = -1.0

        game.step(aL, aR)
        game.render()

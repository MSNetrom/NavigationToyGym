import pygame
import random
import sys
import numpy as np
from abc import ABC, abstractmethod
import gymnasium as gym
from gymnasium import spaces
import math
from pathlib import Path
import json
import matplotlib.pyplot as plt

def draw_arrow(
        surface: pygame.Surface,
        start: pygame.Vector2,
        end: pygame.Vector2,
        color: pygame.Color,
        body_width: int = 2,
        head_width: int = 4,
        head_height: int = 2,
    ):
    """Draw an arrow between start and end with the arrow head at the end.

    Args:
        surface (pygame.Surface): The surface to draw on
        start (numpy.np): Start position
        end (numpy.np): End position
        color (pygame.Color): Color of the arrow
        body_width (int, optional): Defaults to 2.
        head_width (int, optional): Defaults to 4.
        head_height (float, optional): Defaults to 2.
    """
    #print("Drawing arrow:, start:", start, "end:", end)
    arrow = pygame.Vector2(tuple(start - end))
    angle = arrow.angle_to(pygame.Vector2(0, -1))
    body_length = arrow.length() - head_height

    # Create the triangle head around the origin
    head_verts = [
        pygame.Vector2(0, head_height / 2),  # Center
        pygame.Vector2(head_width / 2, -head_height / 2),  # Bottomright
        pygame.Vector2(-head_width / 2, -head_height / 2),  # Bottomleft
    ]
    # Rotate and translate the head into place
    translation = pygame.Vector2(0, arrow.length() - (head_height / 2)).rotate(-angle)
    for i in range(len(head_verts)):
        head_verts[i].rotate_ip(-angle)
        head_verts[i] += translation
        head_verts[i] += start

    pygame.draw.polygon(surface, color, head_verts)

    # Stop weird shapes when the arrow is shorter than arrow head
    if arrow.length() >= head_height:
        # Calculate the body rect, rotate and translate into place
        body_verts = [
            pygame.Vector2(-body_width / 2, body_length / 2),  # Topleft
            pygame.Vector2(body_width / 2, body_length / 2),  # Topright
            pygame.Vector2(body_width / 2, -body_length / 2),  # Bottomright
            pygame.Vector2(-body_width / 2, -body_length / 2),  # Bottomleft
        ]
        translation = pygame.Vector2(0, body_length / 2).rotate(-angle)
        for i in range(len(body_verts)):
            body_verts[i].rotate_ip(-angle)
            body_verts[i] += translation
            body_verts[i] += start

        pygame.draw.polygon(surface, color, body_verts)


# ============================
# Base Classes Provided
# ============================

class InputOutputBlock(ABC):

    @abstractmethod
    def output(self, in_states: np.ndarray | dict[str, np.ndarray], **kwargs) -> np.ndarray:
        pass

class TimeDerivativeFunction(ABC):

    @abstractmethod
    def time_derivative(self, states: np.ndarray | dict[str, np.ndarray]) -> np.ndarray:
        pass
    

class DiscreteStep(InputOutputBlock):

    def __init__(self, system: TimeDerivativeFunction, states_0: np.ndarray, dt: float):
        self.system = system
        try:
            self.states = states_0.copy()
        except:
            self.states = states_0
        self.dt = dt

    @abstractmethod
    def output(self, u: np.ndarray, **kwargs) -> np.ndarray:
        pass

class EulerStep(DiscreteStep):

    def output(self, u: np.ndarray) -> np.ndarray:
        derivatives = self.system.time_derivative({"states": self.states, "u": u})
        self.states += derivatives * self.dt
        return self.states

class HigherOrderRungeKuttaStep(DiscreteStep):

    def output(self, u: np.ndarray) -> np.ndarray:
        k1 = self.dt * self.system.time_derivative(states={"states": self.states, "u": u})
        k2 = self.dt * self.system.time_derivative(states={"states": self.states + k1 / 2, "u": u})
        k3 = self.dt * self.system.time_derivative(states={"states": self.states + k2 / 2, "u": u})
        k4 = self.dt * self.system.time_derivative(states={"states": self.states + k3, "u": u})
        self.states += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return self.states

class Dynamics(ABC):

    def __init__(self, integrator: DiscreteStep):
        self.integrator = integrator
        self._last_u = 0

    def perform_step(self, u: np.ndarray, observation: np.ndarray) -> np.ndarray:
        # Can apply CBF and stuff here
        self._last_u = u
        return self.integrator.output(u)

    def get_used_u(self):
        # Can apply rotation and stuff here
        return self._last_u
    
    @abstractmethod
    def get_initial_states(self) -> np.ndarray:
        pass


class ConstantSpeedObstacle(Dynamics):
    """
    A dynamic obstacle that moves with constant speed.
    The state is assumed to be a 4-element vector: [x, y, vx, vy],
    where (vx, vy) represent the constant velocity.
    """
    class Derivative(TimeDerivativeFunction):
        def time_derivative(self, states: dict[str, np.ndarray]) -> np.ndarray:
            # For a constant-speed obstacle:
            # dx/dt = vx, dy/dt = vy, and no acceleration (dvx/dt = 0, dvy/dt = 0)
            x, y, vx, vy = states["states"]
            return np.array([vx, vy, 0.0, 0.0])
    
    def __init__(self, initial_state: np.ndarray, dt: float, radius: float, color: tuple = (0, 0, 255)):
        """
        Parameters:
            initial_state: np.array([x, y, vx, vy]) describing the obstacle.
            dt: integration time step.
            radius: for collision detection and rendering.
            color: display color.
        """
        self.radius = radius
        self.color = color
        self.initial_state = initial_state
        self.dt = dt
        integrator = EulerStep(ConstantSpeedObstacle.Derivative(), initial_state, dt)
        super().__init__(integrator=integrator)
    
    def get_initial_states(self) -> np.ndarray:
        return self.initial_state
    
    def to_dict(self) -> dict:
        """
        Return a dictionary representation similar to your static obstacles,
        so that the obstacle can be used in collision detection and lidar raycasting.
        """
        pos = self.integrator.states[:2]  # Current position [x, y]
        return {
            'type': 'circle',     # You can choose an appropriate type.
            'pos': pos,
            'vel': self.integrator.states[2:4],
            'radius': self.radius,
            'color': self.color
        }
    

class BaseEnv(ABC):

    @abstractmethod
    def step(self, u: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def reset(self):
        pass

class Env(BaseEnv):

    def __init__(self, dynamics: Dynamics):

        try:
            self.states = dynamics.get_initial_states().copy()
        except:
            self.states = dynamics.get_initial_states()

        self.dynamics = dynamics

    def step(self, u: np.ndarray, observation: np.ndarray) -> np.ndarray:
        self.states = self.dynamics.perform_step(u, observation)
        return self.states

    def reset(self):
        try:
            self.states = self.dynamics.get_initial_states().copy()
        except:
            self.states = self.dynamics.get_initial_states()

        return self.states
    
def load_world(filepath: Path) -> list:
    """
    Load a world configuration from a JSON file.

    Args:
        filepath (str): Path to the JSON file containing world definitions.

    Returns:
        list: A list of obstacle dictionaries.
    """
    with open(filepath, 'r') as file:
        world_data = json.load(file)
    
    obstacles = world_data.get("obstacles", [])
    # Validate and process obstacles if necessary
    return obstacles

# ============================
# Gymnasium-Like Simulation Environment
# ============================

class SimulationEnv(gym.Env):
    """A Gymnasium-like environment with only dynamic obstacles (no static obstacles)."""

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(
        self,
        dynamics: Dynamics,
        render: bool = True,
        border_margin: int = 50,
        num_lidar: int = 16,
        lidar_distance: int = 100,
        u_max: int = None,
        dynamic_obstacles: list = None  # ONLY dynamic obstacles are used now.
    ):
        super(SimulationEnv, self).__init__()

        # Initialize the vehicle environment.
        self.env = Env(dynamics)
        self.render_mode = render
        self.border_margin = border_margin
        self.num_lidar = num_lidar
        self.lidar_distance = lidar_distance

        # Use only dynamic obstacles.
        self.dynamic_obstacles = dynamic_obstacles if dynamic_obstacles is not None else []

        # Rendering setup.
        pygame.init()
        self.WIDTH, self.HEIGHT = 800, 600
        if self.render_mode:
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Gymnasium Simulation Environment")

        # Colors and drawing properties.
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 255, 0)
        self.GRAY = (200, 200, 200)
        self.LIDAR_COLOR = (0, 255, 0)
        self.dot_radius = 0
        self.dot_color = self.RED

        # Define inner boundary—for example, using a rectangle.
        bar_area_width = 100
        self.INNER_RECT = pygame.Rect(
            self.border_margin,
            self.border_margin,
            self.WIDTH - 2 * self.border_margin - bar_area_width,
            self.HEIGHT - 2 * self.border_margin
        )

        self.non_physics_single_frame_objects = []
        self.clock = pygame.time.Clock()
        self.FPS = int(1 / dynamics.integrator.dt)
        self.font = pygame.font.SysFont(None, 24)
        self.bar1_value = 0.0
        self.bar2_value = 0.0
        self.bar1_max = 100.0   # Set a default maximum value for bar1.
        self.bar2_max = 100.0   # Set a default maximum value for bar2.

        # Update the observation space.
        # Here, we now include:
        #   - Vehicle state (assume 6 values: e.g. [x,y,theta,vx,vy,?]) 
        #   - Lidar positions: 2 values per beam.
        #   - Lidar velocities: 1 value per beam.
        # (Adjust the vehicle state size and bounds as necessary.)
        vehicle_low = np.array([border_margin, border_margin, -500.0, -500.0, -math.pi, -500.0], dtype=np.float32)
        vehicle_high = np.array([self.WIDTH - border_margin, self.HEIGHT - border_margin, 500.0, 500.0, math.pi, 500.0], dtype=np.float32)
        lidar_pos_low = np.full(2 * self.num_lidar, -self.lidar_distance, dtype=np.float32)
        lidar_pos_high = np.full(2 * self.num_lidar, self.lidar_distance, dtype=np.float32)
        # For relative velocity, we assume a reasonable bound; adjust as needed.
        lidar_vel_low = np.full(self.num_lidar, -500.0, dtype=np.float32)
        lidar_vel_high = np.full(self.num_lidar, 500.0, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.concatenate((vehicle_low, lidar_pos_low, lidar_vel_low)),
            high=np.concatenate((vehicle_high, lidar_pos_high, lidar_vel_high)),
            dtype=np.float32
        )

    def ray_circle_intersection(self, x, y, ray_angle, circle_x, circle_y, circle_radius):
        """
        Compute and return the distance along the ray (from (x,y) in direction ray_angle)
        to the circle defined by center (circle_x, circle_y) and radius.
        Returns None if there is no intersection in the forward direction.
        """
        dx = math.cos(ray_angle)
        dy = math.sin(ray_angle)
        ox = x - circle_x
        oy = y - circle_y
        A = dx**2 + dy**2
        B = 2 * (dx * ox + dy * oy)
        C = ox**2 + oy**2 - circle_radius**2
        disc = B**2 - 4 * A * C
        if disc < 0:
            return None
        sqrt_disc = math.sqrt(disc)
        t1 = (-B - sqrt_disc) / (2 * A)
        t2 = (-B + sqrt_disc) / (2 * A)
        t_candidates = [t for t in [t1, t2] if t >= 0]
        if not t_candidates:
            return None
        return min(t_candidates)
    

    def ray_wall_intersection(self, x, y, ray_angle):
        """
        Compute the distance from (x,y) along the ray (with angle ray_angle)
        where it intersects the inner boundary (used as the wall).
        Returns None if no intersection is found.
        """
        dx = math.cos(ray_angle)
        dy = math.sin(ray_angle)
        t_values = []
        if not math.isclose(dx, 0, abs_tol=1e-6):
            t_left = (self.INNER_RECT.left - x) / dx
            t_right = (self.INNER_RECT.right - x) / dx
            if t_left >= 0:
                t_values.append(t_left)
            if t_right >= 0:
                t_values.append(t_right)
        if not math.isclose(dy, 0, abs_tol=1e-6):
            t_top = (self.INNER_RECT.top - y) / dy
            t_bottom = (self.INNER_RECT.bottom - y) / dy
            if t_top >= 0:
                t_values.append(t_top)
            if t_bottom >= 0:
                t_values.append(t_bottom)
        if not t_values:
            return None
        return min(t_values)
    


    

    def update_dynamic_obstacles(self):
        """
        Update each dynamic obstacle by stepping its dynamics.
        For constant-speed obstacles, passing a zero control (u = [0, 0]) keeps the speed constant.
        """
        for obs in self.dynamic_obstacles:
            obs.perform_step(u=np.array([0.0, 0.0]), observation=None)

    def check_collision(self, state: np.ndarray) -> tuple[bool, float]:
        x, y, *_ = state

        # Check collision with all dynamic obstacles only.
        for dyn_obs in self.dynamic_obstacles:
            obs_pos = dyn_obs.integrator.states[:2]
            dx = x - obs_pos[0]
            dy = y - obs_pos[1]
            dist = math.hypot(dx, dy)
            if dist < dyn_obs.radius:
                penetration_depth = dyn_obs.radius - dist  # positive if overlapping
                return True, penetration_depth

        # Check the inner boundary as a wall.
        if not self.INNER_RECT.collidepoint(x, y):
            breach_depths = []
            if x < self.INNER_RECT.left:
                breach_depths.append(self.INNER_RECT.left - x)
            elif x > self.INNER_RECT.right:
                breach_depths.append(x - self.INNER_RECT.right)
            if y < self.INNER_RECT.top:
                breach_depths.append(self.INNER_RECT.top - y)
            elif y > self.INNER_RECT.bottom:
                breach_depths.append(y - self.INNER_RECT.bottom)
            if breach_depths:
                breach_depth = min(breach_depths)
                return True, breach_depth

        return False, 0.0

    def get_lidar_sensor_readings(self):
        """
        Compute two sorts of lidar quantities:
         1. The relative positions of the beam intersection points in the vehicle's LiDAR frame.
         2. The full relative 2D velocity vector for the beam intersection point,
            computed as (obstacle_vel - vehicle_vel) in the vehicle's frame.

        Assumes:
         - Vehicle state structure: [x, y, theta, vx, vy, ...]
         - Each dynamic obstacle's to_dict() returns a dictionary that includes:
             'pos': [ox, oy] and 'vel': [ovx, ovy]
        """
        state = self.env.states
        # Unpack vehicle state: (x, y, theta, vx, vy, ...). Adjust indices if needed.
        x, y, theta, vx, vy = state[:5]
        # Allocate positions and velocities in the vehicle (lidar) frame.
        lidar_positions = np.zeros((self.num_lidar, 2), dtype=np.float32)
        lidar_velocities = np.zeros((self.num_lidar, 2), dtype=np.float32)
        # Compute beam angles in the vehicle frame.
        angles = np.linspace(0, 2 * math.pi, self.num_lidar, endpoint=False)
        
        for i, lidar_angle in enumerate(angles):
            world_angle = lidar_angle + theta
            max_distance = self.lidar_distance
            min_distance = max_distance
            hit_obstacle = None  # will store the dictionary of the obstacle that is hit (if any)

            # Check all dynamic obstacles.
            for obs in self.dynamic_obstacles:
                obs_dict = obs.to_dict()  # Assumes obs_dict contains 'pos', 'radius', and 'vel'
                d = self.ray_circle_intersection(
                    x, y, world_angle,
                    obs_dict['pos'][0], obs_dict['pos'][1],
                    obs_dict['radius']
                )
                if d is not None and d < min_distance:
                    min_distance = d
                    hit_obstacle = obs_dict

            # Also check for wall intersection.
            wall_distance = self.ray_wall_intersection(x, y, world_angle)
            if wall_distance is not None and wall_distance < min_distance:
                min_distance = wall_distance
                hit_obstacle = None  # wall is considered static

            # Compute the relative position (in the vehicle's LiDAR frame).
            # Note: lidar_angle is already in the vehicle frame.
            lidar_positions[i, 0] = min_distance * math.cos(lidar_angle)
            lidar_positions[i, 1] = min_distance * math.sin(lidar_angle)

            # Compute the full 2D relative velocity.
            if hit_obstacle is not None and 'vel' in hit_obstacle:
                obs_vel = np.array(hit_obstacle['vel'])      # Obstacle velocity in world frame.
                vehicle_vel = np.array([vx, vy])               # Vehicle velocity in world frame.
                rel_vel_world = obs_vel - vehicle_vel           # Relative velocity in world frame.
                # Rotate relative velocity into the vehicle (lidar) frame.
                # The rotation by -theta converts world-frame vectors to vehicle frame.
                cos_th = math.cos(theta)
                sin_th = math.sin(theta)
                R_inv = np.array([[cos_th, sin_th],
                                  [-sin_th, cos_th]])
                rel_vel_vehicle = R_inv @ rel_vel_world
                lidar_velocities[i, :] = rel_vel_vehicle
            else:
                # For the wall (or no hit), assume relative velocity to be based of the vehicle's velocity
                lidar_velocities[i, :] = - np.array([0, 0])
        
        return lidar_positions, lidar_velocities

    def step(self, action: np.ndarray, observation: np.ndarray, i):
        """
        Perform one simulation step:
           - First update the dynamic obstacles.
           - Then update the vehicle state.
           - Finally, compute the sensor measurements.
        """
        self.update_dynamic_obstacles()  # update obstacles

        new_state = self.env.step(action, observation)
        # Get both lidar positions and relative velocities.
        lidar_positions, lidar_velocities = self.get_lidar_sensor_readings()
        reward = 0.0
        crash, dist = self.check_collision(new_state)
        info = {"crash": crash, "crash_distance": dist}
        # Build observation as:
        # [vehicle_state, lidar_positions.flatten(), lidar_velocities.flatten()]
        observation = np.concatenate((new_state, lidar_positions.flatten(), lidar_velocities.flatten())).astype(np.float32)
        return observation, reward, False, info

    def reset(self):
        """
        Reset the vehicle and return initial observation, including sensor readings.
        """
        self.env.reset()
        initial_state = self.env.states.copy()
        lidar_positions, lidar_velocities = self.get_lidar_sensor_readings()
        observation = np.concatenate((initial_state, lidar_positions.flatten(), lidar_velocities.flatten())).astype(np.float32)
        return observation, {}

    def draw_figure(self, fig):

        if fig['type'] == 'circle':
            pygame.draw.circle(self.screen, fig['color'],
                                (int(fig['pos'][0]), int(fig['pos'][1])), fig['radius'])
        elif fig['type'] == 'rectangle':
            rect = pygame.Rect(fig['pos'][0], fig['pos'][1],
                                fig['width'], fig['height'])
            pygame.draw.rect(self.screen, fig['color'], rect)
        elif fig['type'] == 'arrow':
            draw_arrow(self.screen, fig['start'], fig['end'], fig['color'], body_width=10, head_width=20, head_height=20)


    def render(self):
        """
        Render the environment.
        """
        if not self.render_mode:
            return

        # Fill the background
        self.screen.fill(self.WHITE)

        # Draw outer window boundary
        pygame.draw.rect(self.screen, self.BLACK, self.screen.get_rect(), 2)

        # Draw inner boundary (walls are passable)
        pygame.draw.rect(self.screen, self.BLACK, self.INNER_RECT, 2)

        # Draw dynamic obstacles.
        for dyn_obs in self.dynamic_obstacles:
            self.draw_figure(dyn_obs.to_dict())

        # Draw any single-frame non-physics objects.
        for fig in self.non_physics_single_frame_objects:
            self.draw_figure(fig)
        
        self.non_physics_single_frame_objects.clear()

        # Draw the dot/car
        x, y, theta, *_ = self.env.states
        #dot_center = (int(x), int(y))

        # Draw rotated rectangle to indicate orientation
        #rect_length = 40  # Increased length for better visibility
        #rect_width = 20
        #rect = pygame.Surface((rect_length, rect_width), pygame.SRCALPHA)
        #rect.fill(self.dot_color)
        #rotated_rect = pygame.transform.rotate(rect, -math.degrees(theta))
        #rect_rect = rotated_rect.get_rect(center=dot_center)
        #self.screen.blit(rotated_rect, rect_rect.topleft)

        # Draw lidar lines
        self.draw_lidar()

        # Display velocity and orientation
        #vel_text = self.font.render(f"Velocity: ({vx:.2f}, {vy:.2f})", True, self.BLACK)
        #self.screen.blit(vel_text, (10, 10))
        theta_deg = math.degrees(theta) % 360
        theta_text = self.font.render(f"Orientation: {theta_deg:.2f} degrees", True, self.BLACK)
        self.screen.blit(theta_text, (10, 30))

        # Display instructions
        instr_text1 = self.font.render("Press C/R to add obstacles. ESC to exit.", True, self.BLACK)
        instr_text2 = self.font.render(f"Lidar Beams: {self.num_lidar}", True, self.BLACK)
        self.screen.blit(instr_text1, (10, self.HEIGHT - 40))
        self.screen.blit(instr_text2, (10, self.HEIGHT - 20))

        # ============================
        # Draw Acceleration Bars
        # ============================

        # Define bar properties with increased dimensions and spacing
        bar_width = 25  # Increased from 15 to 25
        bar_spacing = 15  # Increased from 5 to 15
        bar_max_height = 300  # Increased from 200 to 300 pixels
        bar_outline_thickness = 2

        # Additional spacing between the bars and the game area, and the window's right edge
        bar_side_margin = 20  # Increased from 10 to 20 pixels

        # Calculate available space for bars
        bar_area_x_start = self.INNER_RECT.right + bar_side_margin
        bar_y = self.HEIGHT - self.border_margin - bar_max_height - 50  # 50 pixels above the bottom

        # Positions for the two bars
        bar1_x = bar_area_x_start
        bar2_x = bar1_x + bar_width + bar_spacing

        # Draw Bar 1 Outline
        pygame.draw.rect(self.screen, self.BLACK, (bar1_x, bar_y, bar_width, bar_max_height), bar_outline_thickness)

        # Calculate Bar 1 Filled Height
        bar1_fill_height = (self.bar1_value / self.bar1_max) * bar_max_height
        bar1_fill_height = max(0, min(bar_max_height, bar1_fill_height))  # Clamp to [0, bar_max_height]

        #print((bar2_x, bar_y, bar_width, bar_max_height))

        # Draw Bar 1 Filled
        pygame.draw.rect(self.screen, self.BLUE, 
                         (int(bar1_x), int(bar_y + bar_max_height - bar1_fill_height), int(bar_width), int(bar1_fill_height)))

        # Draw Bar 2 Outline
        pygame.draw.rect(self.screen, self.BLACK, (int(bar2_x), int(bar_y), int(bar_width), int(bar_max_height)), bar_outline_thickness)

        # Calculate Bar 2 Filled Height
        bar2_fill_height = (self.bar2_value / self.bar2_max) * bar_max_height
        bar2_fill_height = max(0, min(bar_max_height, bar2_fill_height))  # Clamp to [0, bar_max_height]

        # Draw Bar 2 Filled
        pygame.draw.rect(self.screen, self.GREEN, 
                         (int(bar2_x), int(bar_y + bar_max_height - bar2_fill_height), int(bar_width), int(bar2_fill_height)))

        # Add Labels for Bars
        label1 = self.font.render("Ax", True, self.BLACK)
        label2 = self.font.render("Ay", True, self.BLACK)
        label1_width, label1_height = self.font.size("Ax")
        label2_width, label2_height = self.font.size("Ay")

        # Center the labels below the bars
        self.screen.blit(label1, (bar1_x + (bar_width - label1_width) / 2, bar_y + bar_max_height + 5))
        self.screen.blit(label2, (bar2_x + (bar_width - label2_width) / 2, bar_y + bar_max_height + 5))

        # Optional: Display numerical values above the bars
        value1_text = self.font.render(f"{self.bar1_value:.1f}", True, self.BLACK)
        value2_text = self.font.render(f"{self.bar2_value:.1f}", True, self.BLACK)
        value1_width, _ = self.font.size(f"{self.bar1_value:.1f}")
        value2_width, _ = self.font.size(f"{self.bar2_value:.1f}")

        # Center the values above the bars
        self.screen.blit(value1_text, (bar1_x + (bar_width - value1_width) / 2, bar_y - 25))
        self.screen.blit(value2_text, (bar2_x + (bar_width - value2_width) / 2, bar_y - 25))

        # Update the display
        pygame.display.flip()

        self.clock.tick(self.FPS)

    def render_final_figures(self, fig_list, path: str):
        """
        Render the final list of figures and save the image.
        """

        # Draw additional figures
        for fig in fig_list:
            self.draw_figure(fig)

        # Update the display to show final figures
        pygame.display.flip()

        # Save the final image
        self.save_image(path)

    def save_image(self, filepath: str):
        """Save the current Pygame screen as a PNG image."""
        pygame.image.save(self.screen, filepath)
        print(f"Saved simulation snapshot as {filepath}")

    def close(self):
        """
        Perform any necessary cleanup.
        """
        if self.render_mode:
            pygame.quit()

    def add_single_frame_non_physics_object(self, obj_specs):
        """
        Add a single frame non-physics object to the environment.

        Parameters:
            obj_specs: {'type': 'circle', 'pos': pos, 'radius': radius, 'color': color}
                       {'type': 'rectangle', 'pos': pos, 'width': width, 'height': height, 'color': color}
                       {'type': 'arrow', 'start': start, 'end': end, 'color': color},
                       {'type': 'ellipse', 'pos': pos, 'a_vec': a_vec, 'b_vec': b_vec, 'color': color}

        Returns:
            None
        """
        self.non_physics_single_frame_objects.append(obj_specs)

    def draw_lidar(self):
        """
        Draw lidar beams and intersection points.
        Uses the sensor method to obtain the intersection positions.
        """
        # Get the sensor readings (relative positions and velocities; we use positions for drawing).
        positions, _ = self.get_lidar_sensor_readings()  # positions shape: (num_lidar, 2)

        # Retrieve the vehicle state: [x, y, theta, ...].
        state = self.env.states
        x, y, theta = state[:3]
        cos_th = math.cos(theta)
        sin_th = math.sin(theta)

        # Build the 2x2 rotation matrix to transform from vehicle frame to world frame.
        rotation_matrix = np.array([[cos_th, -sin_th],
                                    [sin_th,  cos_th]])

        for i in range(self.num_lidar):
            # Relative position from the lidar sensor (in the vehicle frame).
            rel_vector = positions[i]  # A (2,) vector.
            # Transform the relative vector to the world frame.
            world_vector = rotation_matrix @ rel_vector
            world_coords = np.array([x, y]) + world_vector

            # Draw the beam from the vehicle's position to the intersection point.
            pygame.draw.line(self.screen, self.LIDAR_COLOR,
                             (int(x), int(y)),
                             (int(world_coords[0]), int(world_coords[1])),
                             1)
            # Draw the intersection point as a small circle.
            pygame.draw.circle(self.screen, self.LIDAR_COLOR,
                               (int(world_coords[0]), int(world_coords[1])),
                               2)

    def render_frame(self):
        """
        Render the current frame.
        """
        self.render()

    def process_event(self, event):
        """
        Process Pygame events for obstacle addition.

        Parameters:
            event (pygame.event.Event): The event to process.
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.close()
                sys.exit()

    # ============================
    # Added Method to Set Bar Values
    # ============================

    def set_bars(self, val1, val2):
        """
        Set the values of the two acceleration indicator bars.

        Parameters:
            val1 (float): Value for the first bar (e.g., ax).
            val2 (float): Value for the second bar (e.g., ay).

        Returns:
            None
        """
        # Clip the values to the action space limits for visual consistency
        self.bar1_value = val1
        self.bar2_value = val2


    


class BicycleCarDynamics(Dynamics):

    class Derivative(TimeDerivativeFunction):

        def __init__(self, length=2.5, rotation_damping=0.0005, linear_friction=0.1):
            self.length = length
            self.rotation_damping = rotation_damping
            self.linear_friction = linear_friction

        def time_derivative(self, states: dict[str, np.ndarray]) -> np.ndarray:
            # states["states"] = [x, y, vx, vy, theta, omega]
            # states["u"] = [ax, ay, alpha] (alpha is steering angle)
            x, y, theta, vx, vy, omega = states["states"]
            ax, ay, alpha = states["u"].flatten()

            # Apply linear friction (Friction force opposite to velocity)
            friction_x = -self.linear_friction * vx
            friction_y = -self.linear_friction * vy

            # Steering angle is limited to prevent extreme steering
            max_steering_angle = math.radians(30)  # 30 degrees limit
            steering_angle = max(-max_steering_angle, min(max_steering_angle, alpha))

            # Calculate angular acceleration based on steering angle
            if abs(steering_angle) > 1e-4 and abs(vx) > 1e-4:
                turning_radius = self.length / math.tan(steering_angle)
                angular_acceleration = vx / turning_radius
            else:
                angular_acceleration = 0.0

            # Update angular velocity with steering and damping
            domega_dt = angular_acceleration - self.rotation_damping * omega

            # Update orientation based on angular velocity
            dtheta_dt = omega

            # Update positions based on current velocity and orientation
            dx_dt = vx * math.cos(theta) - vy * math.sin(theta)
            dy_dt = vx * math.sin(theta) + vy * math.cos(theta)

            # Update velocities
            dvx_dt = ax + friction_x
            dvy_dt = 0.0 + friction_y  # No lateral acceleration for simplicity

            return np.array([dx_dt, dy_dt, dtheta_dt, dvx_dt, dvy_dt, domega_dt])



    def __init__(self, control_size, initial_state, dt, length=2.5, rotation_damping=0.0005, linear_friction=0.1):
        """
        length: Distance between the front and rear axles.
        rotation_damping: Factor to slow down rotation. Lower values mean slower rotation.
        linear_friction: Coefficient for linear friction.
        """
        super().__init__(integrator=HigherOrderRungeKuttaStep(BicycleCarDynamics.Derivative(
            length=length, rotation_damping=rotation_damping, linear_friction=linear_friction
        ), initial_state, dt))
        self.control_size = control_size
        self.initial_state = initial_state
        self.length = length
        self.rotation_damping = rotation_damping
        self.linear_friction = linear_friction

    def get_initial_states(self) -> np.ndarray:
        return self.initial_state
    
    def get_used_u(self):
        # Get current alpha value
        alpha = self.integrator.states[2]
        # Rotate u by this angle
        R = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
        u = R @ self._last_u[:2]
        return u

    def map_keys_to_actions(self, keys: list[bool]) -> np.ndarray:
        """
        Map key presses to action vector for BicycleCarDynamics.
        [ax, ay, alpha]
        """
        ax = 0.0
        ay = 0.0  # Unused
        alpha = 0.0

        if keys[pygame.K_UP]:
            ax += self.control_size  # Forward acceleration
        if keys[pygame.K_DOWN]:
            ax -= self.control_size  # Backward acceleration

        # Steering without accumulation
        if keys[pygame.K_LEFT] and not keys[pygame.K_RIGHT]:
            alpha = -math.radians(5)  # Negative steering angle
        elif keys[pygame.K_RIGHT] and not keys[pygame.K_LEFT]:
            alpha = math.radians(5)  # Positive steering angle

        return np.array([ax, ay, alpha], dtype=np.float32)


class DotDynamicsNormal(Dynamics):

    class Derivative(TimeDerivativeFunction):

        def _f(self, states: np.ndarray) -> np.ndarray:
            x, y, theta, v_x, v_y = states
            return np.array([v_x, v_y, 0.0, 0.0, 0.0])
    

        def _G(self, states: np.ndarray) -> np.ndarray:
            return np.array([[0, 0], 
                            [0, 0], 
                            [0, 0], 
                            [1, 0],
                            [0, 1]])

        def time_derivative(self, states: dict[str, np.ndarray]) -> np.ndarray:
            return self._f(states["states"]) + self._G(states["states"]) @ states["u"].flatten()



    def __init__(self, dt=1e-2, initial_state = np.array([400, 300, 0.0, 0.0, 0.0], dtype=float), control_size=50,
                 constant_control: np.ndarray = None):
        super().__init__(integrator=HigherOrderRungeKuttaStep(DotDynamicsNormal.Derivative(), initial_state, dt))
        self.control_size = control_size
        self.dt = dt
        self.initial_state = initial_state
        self.constant_control = constant_control

    def get_initial_states(self) -> np.ndarray:
        return self.initial_state
    

    def map_keys_to_actions(self, keys: list[bool]) -> np.ndarray:
        """
        Map key presses to action vector for DotDynamics.
        [ax, ay, alpha] where alpha is unused.
        """
        if self.constant_control is not None:
            return self.constant_control

        ax = 0.0
        ay = 0.0
        K = self.control_size

        if keys[pygame.K_LEFT]:
            ax -= K  # Accelerate left
        if keys[pygame.K_RIGHT]:
            ax += K  # Accelerate right
        if keys[pygame.K_UP]:
            ay -= K  # Accelerate up
        if keys[pygame.K_DOWN]:
            ay += K # Accelerate down

        return np.array([ax, ay], dtype=np.float32)


def runner(dynamics: Dynamics, lidar_distance: float, lidar_num: int, u_max: float, render: bool,
           num_steps: int = 1000, results_path: Path = None, dt: float = None,
           dynamic_obstacles: list = None):
    
    sim_env = SimulationEnv(
         dynamics=dynamics,
         render=render,
         border_margin=50,
         num_lidar=lidar_num,
         lidar_distance=lidar_distance,
         u_max=u_max,
         dynamic_obstacles=dynamic_obstacles  # Pass only dynamic obstacles.
    )

    observation, _ = sim_env.reset()
    observation_track = np.zeros((num_steps, observation.shape[0]))
    u_ref_track = np.zeros((num_steps, 2))
    u_actual_track = np.zeros((num_steps, 2))

    for i in range(num_steps):
        # Process Pygame events
        for event in pygame.event.get():
            sim_env.process_event(event)

        keys = pygame.key.get_pressed()
        u = dynamics.map_keys_to_actions(keys)
        observation, _, done, info = sim_env.step(u, observation, i)

        u_used = dynamics.get_used_u()
        # (Optional: add arrows or other illustrations)
        sim_env.add_single_frame_non_physics_object({
            'type': 'arrow',
            'start': observation[:2],
            'end': observation[:2] + u[:2],
            'color': (255, 120, 120)
        })
        sim_env.add_single_frame_non_physics_object({
            'type': 'arrow',
            'start': observation[:2],
            'end': observation[:2] + u_used,
            'color': (50, 205, 50)
        })

        sim_env.render()
        sim_env.set_bars(abs(u_used[0]), abs(u_used[1]))
        observation_track[i] = observation
        u_ref_track[i] = u
        u_actual_track[i] = u_used

    # (Plotting and saving results remain unchanged.)
    ...

    sim_env.close()

if __name__ == "__main__":
    U_MAX = 50
    LIDAR_BEAMS = 64

    # Example: Using your soft-min CBF controlled vehicle dynamics (from soft_min_nav_linear.py)
    from soft_min_nav_linear import DotDynamicsSpeedSoftMin, DotDynamicsNormalSoftMin
    dynamics = DotDynamicsSpeedSoftMin(dt=1e-2, radius=10, u_max=U_MAX, p1=3, p2=2.01, k=0.1,
                                        lidar_num=LIDAR_BEAMS,
                                        initial_state=np.array([200.0, 300.0, 0.0, 20.0, 0.0]))
    
    #dynamics = DotDynamicsNormalSoftMin(dt=1e-2, radius=0.1, u_max=U_MAX, p1=3, p2=2.01, k=0.1,
    #                                    initial_state=np.array([200.0, 300.0, 0.0, 20.0, 0.0]))

    # Create dynamic obstacles.
    # For "static" obstacles, simply set their speed to zero.
    #from dynamic_obstacle import ConstantSpeedObstacle
    dynamic_obs1 = ConstantSpeedObstacle(
         initial_state=np.array([400.0, 300.0, -20.0, 0.0]),  # Speed set to zero = static
         dt=1e-2,
         radius=15,
         color=(0, 0, 255)
    )
    dynamic_obs2 = ConstantSpeedObstacle(
         initial_state=np.array([400.0, 300.0, 0.0, -100.0]),
         dt=1e-2,
         radius=20,
         color=(255, 0, 0)
    )

    # Run simulation with only dynamic obstacles.
    runner(
         dynamics=dynamics,
         lidar_distance=130,
         lidar_num=LIDAR_BEAMS,
         u_max=U_MAX,
         render=True,
         #results_path=Path("results"),
         num_steps=5000,
         dynamic_obstacles=[dynamic_obs1]
    )
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
    """A Gymnasium-like environment encapsulating a Pygame simulation."""

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, dynamics: Dynamics,
                 render: bool = True, border_margin: int = 50,
                 num_lidar: int = 16, lidar_distance: int = 100,
                 initial_obstacles: int = 10, u_max: int = None,
                 world_file: Path = None):
        
        super(SimulationEnv, self).__init__()

        # Initialize the base environment
        self.env = Env(dynamics)

        self.render_mode = render
        self.border_margin = border_margin
        self.num_lidar = num_lidar
        self.lidar_distance = lidar_distance
        self.initial_obstacles = initial_obstacles

        # Observation: [x, y, vx, vy, theta, omega, lidar_relative_vectors_flattened]
        lidar_low = np.full(2 * self.num_lidar, -self.lidar_distance, dtype=np.float32)
        lidar_high = np.full(2 * self.num_lidar, self.lidar_distance, dtype=np.float32)

        self.observation_space = spaces.Box(
            low=np.concatenate((
                np.array([border_margin, border_margin, -500.0, -500.0, -math.pi, -500.0], dtype=np.float32),
                lidar_low
            )),
            high=np.concatenate((
                np.array([800 - border_margin, 600 - border_margin, 500.0, 500.0, math.pi, 500.0], dtype=np.float32),
                lidar_high
            )),
            dtype=np.float32
        )

        # Rendering setup
        pygame.init()

        # Screen dimensions
        self.WIDTH, self.HEIGHT = 800, 600

        if self.render_mode:
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
            pygame.display.set_caption("Gymnasium Simulation Environment")

        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 255, 0)
        self.GRAY = (200, 200, 200)
        self.LIDAR_COLOR = (0, 255, 0)  # Green color for lidar lines

        # Dot properties
        self.dot_radius = 0
        self.dot_color = self.RED

        # Inner boundary properties
        bar_area_width = 100  # Space allocated for the bars on the right
        self.INNER_RECT = pygame.Rect(
            self.border_margin,
            self.border_margin,
            self.WIDTH - 2 * self.border_margin - bar_area_width,
            self.HEIGHT - 2 * self.border_margin
        )

        # Non-physics single frame objects
        self.non_physics_single_frame_objects = []

        # Obstacles
        self.obstacles = []  # List to store obstacles

        self.use_world_file = world_file is not None

        if self.use_world_file:
            print(f"Loading obstacles from file: {world_file}")
            self.load_obstacles_from_file(world_file)
        else:
            print("No world file provided.")
            # Initialize with given number of random obstacles
            for _ in range(self.initial_obstacles):
                self.add_random_obstacle()

        # Clock for controlling frame rate
        self.clock = pygame.time.Clock()
        self.FPS = int(1 / dynamics.integrator.dt)

        # Font for displaying text
        self.font = pygame.font.SysFont(None, 24)

        # ============================
        # Added Variables for Acceleration Bars
        # ============================
        self.bar1_value = 0.0  # First acceleration input (e.g., Ax)
        self.bar2_value = 0.0  # Second acceleration input (e.g., Ay)

        # Define maximum values for scaling bars
        self.bar1_max = u_max if u_max is not None else 1.0
        self.bar2_max = u_max if u_max is not None else 1.0

    def load_obstacles_from_file(self, filepath: str):
        """
        Load obstacles from a JSON file.

        Args:
            filepath (str): Path to the JSON file.
        """
        loaded_obstacles = load_world(filepath)
        for obj in loaded_obstacles:
            if obj['type'] == 'circle':
                self.obstacles.append({
                    'type': 'circle',
                    'pos': np.array(obj['pos'], dtype=float),
                    'radius': obj['radius'],
                    'color': tuple(obj['color'])
                })
            elif obj['type'] == 'rectangle':
                self.obstacles.append({
                    'type': 'rectangle',
                    'pos': np.array(obj['pos'], dtype=float),
                    'width': obj['width'],
                    'height': obj['height'],
                    'color': tuple(obj['color'])
                })
            else:
                print(f"Unknown obstacle type: {obj['type']}")

    def step(self, action: np.ndarray, observation: np.ndarray):
        """
        Perform one step in the environment.

        Parameters:
            action (np.ndarray): Acceleration input [ax, ay, alpha].

        Returns:
            observation (np.ndarray): The next observation.
            reward (float): Reward for the action (not implemented, set to 0).
            done (bool): Whether the episode has ended (not implemented, set to False).
            info (dict): Additional information (empty).
        """

        # Apply acceleration
        u = action  # Shape (3,) for [ax, ay, alpha]

        # Step the physics
        new_state = self.env.step(u, observation)

        # Compute lidar relative vectors
        lidar_rel_vectors = self.get_lidar_relative_vectors()

        # No reward structure defined; set reward to 0
        reward = 0.0

        # Episode never ends in this setup
        done = False

        # No additional info
        info = {}

        # Observation includes state and lidar relative vectors
        observation = np.concatenate((new_state, lidar_rel_vectors.flatten())).astype(np.float32)

        return observation, reward, done, info

    def reset(self):
        """
        Reset the environment to an initial state and return the initial observation.

        Returns:
            observation (np.ndarray): The initial observation.
            info (dict): Additional information (empty).
        """
        self.env.reset()
        initial_state = self.env.states.copy()

        # Reset obstacles only if not using a world file
        if not self.use_world_file:
            self.obstacles.clear()
            for _ in range(self.initial_obstacles):
                self.add_random_obstacle()

        # Compute initial lidar relative vectors
        lidar_rel_vectors = self.get_lidar_relative_vectors()

        # Reset bar values
        self.bar1_value = 0.0
        self.bar2_value = 0.0

        # Observation includes state and lidar relative vectors
        observation = np.concatenate((initial_state, lidar_rel_vectors.flatten())).astype(np.float32)

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

        # Draw obstacles, and non-detectable shapes
        for obstacle in self.obstacles + self.non_physics_single_frame_objects:
            self.draw_figure(obstacle)
            
            #elif obstacle['type'] == 'ellipse':
            #    ellipse_rect = pygame.Rect(obstacle['pos'][0], obstacle['pos'][1],
                                       

        self.non_physics_single_frame_objects.clear()

        # Draw the dot/car
        x, y, theta, *_ = self.env.states
        dot_center = (int(x), int(y))

        # Draw rotated rectangle to indicate orientation
        rect_length = 40  # Increased length for better visibility
        rect_width = 20
        rect = pygame.Surface((rect_length, rect_width), pygame.SRCALPHA)
        rect.fill(self.dot_color)
        rotated_rect = pygame.transform.rotate(rect, -math.degrees(theta))
        rect_rect = rotated_rect.get_rect(center=dot_center)
        self.screen.blit(rotated_rect, rect_rect.topleft)

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
        instr_text2 = self.font.render(f"Lidar Beams: {self.num_lidar}, Initial Obstacles: {self.initial_obstacles}", True, self.BLACK)
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

    def add_random_obstacle(self):
        """Add a random obstacle (circle or rectangle) within the inner boundary."""
        obstacle_type = random.choice(['circle', 'rectangle'])
        if obstacle_type == 'circle':
            radius = random.randint(10, 30)
            pos = np.array([
                random.randint(self.INNER_RECT.left + radius, self.INNER_RECT.right - radius),
                random.randint(self.INNER_RECT.top + radius, self.INNER_RECT.bottom - radius)
            ], dtype=float)
            color = (
                random.randint(50, 255),
                random.randint(50, 255),
                random.randint(50, 255)
            )
            self.obstacles.append({'type': 'circle', 'pos': pos, 'radius': radius, 'color': color})
        else:
            width = random.randint(20, 60)
            height = random.randint(20, 60)
            pos = np.array([
                random.randint(self.INNER_RECT.left, self.INNER_RECT.right - width),
                random.randint(self.INNER_RECT.top, self.INNER_RECT.bottom - height)
            ], dtype=float)
            color = (
                random.randint(50, 255),
                random.randint(50, 255),
                random.randint(50, 255)
            )
            self.obstacles.append({'type': 'rectangle', 'pos': pos, 'width': width, 'height': height, 'color': color})

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

    def get_lidar_relative_vectors(self):
        """
        Compute lidar-like relative vectors in the agent's coordinate frame.

        Returns:
            np.ndarray: Relative vectors [ [dx1, dy1], [dx2, dy2], ..., [dxN, dyN] ].
        """
        x, y, theta, *_ = self.env.states
        lidar_rel_vectors = np.zeros((self.num_lidar, 2), dtype=np.float32)
        angles = np.linspace(0, 2 * math.pi, self.num_lidar, endpoint=False)

        for i, lidar_angle in enumerate(angles):
            # Compute the absolute angle in the world frame
            world_angle = lidar_angle + theta

            # Define maximum lidar distance
            max_distance = self.lidar_distance

            # Ray casting to find the nearest intersection
            min_dist = max_distance
            for obstacle in self.obstacles:
                if obstacle['type'] == 'circle':
                    dist = self.ray_circle_intersection(x, y, world_angle, obstacle['pos'][0], obstacle['pos'][1], obstacle['radius'] + self.dot_radius)
                    if dist is not None and dist < min_dist:
                        min_dist = dist
                elif obstacle['type'] == 'rectangle':
                    dist = self.ray_rectangle_intersection(x, y, world_angle, obstacle)
                    if dist is not None and dist < min_dist:
                        min_dist = dist

            # Also check intersection with walls
            wall_dist = self.ray_wall_intersection(x, y, world_angle)
            if wall_dist is not None and wall_dist < min_dist:
                min_dist = wall_dist

            # Compute the relative vector in the agent's frame
            rel_x = min_dist * math.cos(lidar_angle)
            rel_y = min_dist * math.sin(lidar_angle)
            lidar_rel_vectors[i] = [rel_x, rel_y]

        return lidar_rel_vectors

    def ray_circle_intersection(self, x, y, angle, cx, cy, radius):
        """
        Calculate the distance from (x, y) in the given angle to the circle at (cx, cy) with radius.

        Returns:
            float or None: Distance to the intersection point, or None if no intersection.
        """
        dx = math.cos(angle)
        dy = math.sin(angle)
        fx = x - cx
        fy = y - cy

        a = dx**2 + dy**2
        b = 2 * (fx * dx + fy * dy)
        c = fx**2 + fy**2 - radius**2

        discriminant = b**2 - 4 * a * c

        if discriminant < 0:
            return None  # No intersection

        discriminant = math.sqrt(discriminant)

        t1 = (-b - discriminant) / (2 * a)
        t2 = (-b + discriminant) / (2 * a)

        if t1 >= 0:
            return t1 * math.sqrt(a)
        if t2 >= 0:
            return t2 * math.sqrt(a)
        return None

    def ray_rectangle_intersection(self, x, y, angle, obstacle):
        """
        Calculate the distance from (x, y) in the given angle to the rectangle obstacle.

        Returns:
            float or None: Distance to the intersection point, or None if no intersection.
        """
        dx = math.cos(angle)
        dy = math.sin(angle)
        rect = pygame.Rect(obstacle['pos'][0], obstacle['pos'][1], obstacle['width'], obstacle['height'])

        # Define rectangle edges
        edges = [
            ((rect.left, rect.top), (rect.right, rect.top)),     # Top
            ((rect.right, rect.top), (rect.right, rect.bottom)), # Right
            ((rect.right, rect.bottom), (rect.left, rect.bottom)), # Bottom
            ((rect.left, rect.bottom), (rect.left, rect.top))    # Left
        ]

        min_dist = self.lidar_distance
        for edge in edges:
            pt1, pt2 = edge
            intersect = self.ray_line_intersection(x, y, dx, dy, pt1[0], pt1[1], pt2[0], pt2[1])
            if intersect is not None:
                dist = math.hypot(intersect[0] - x, intersect[1] - y)
                if dist < min_dist:
                    min_dist = dist

        if min_dist < self.lidar_distance:
            return min_dist
        return None

    def ray_wall_intersection(self, x, y, angle):
        """
        Calculate the distance from (x, y) in the given angle to the walls (inner boundary).

        Returns:
            float or None: Distance to the intersection point, or None if no intersection.
        """
        # Walls are the edges of INNER_RECT treated as a rectangle obstacle
        rect = self.INNER_RECT
        return self.ray_rectangle_intersection(x, y, angle, {
            'type': 'rectangle',
            'pos': np.array([rect.left, rect.top]),
            'width': rect.width,
            'height': rect.height,
            'color': self.BLACK
        })

    def ray_line_intersection(self, x, y, dx, dy, x1, y1, x2, y2):
        """
        Calculate the intersection point of a ray and a line segment.

        Returns:
            tuple or None: (ix, iy) intersection point, or None if no intersection.
        """
        # Ray: (x, y) + t*(dx, dy), t >= 0
        # Segment: (x1, y1) to (x2, y2)
        denominator = (dx * (y1 - y2) - dy * (x1 - x2))
        if denominator == 0:
            return None  # Parallel

        t = ((x1 - x) * (y1 - y2) - (y1 - y) * (x1 - x2)) / denominator
        u = ((x - x1) * dy - (y - y1) * dx) / denominator

        if t >= 0 and 0 <= u <= 1:
            ix = x + t * dx
            iy = y + t * dy
            return (ix, iy)
        return None

    def draw_lidar(self):
        """
        Draw lidar lines on the screen.
        """
        x, y, theta, *_ = self.env.states
        lidar_rel_vectors = self.get_lidar_relative_vectors()

        # Rotate relative vectors to world frame for drawing
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

        for rel_vector in lidar_rel_vectors:
            world_vector = rotation_matrix @ rel_vector
            end_x = x + world_vector[0]
            end_y = y + world_vector[1]
            pygame.draw.line(self.screen, self.LIDAR_COLOR, (int(x), int(y)), (int(end_x), int(end_y)), 1)
            # Draw small circles at the end points
            pygame.draw.circle(self.screen, self.LIDAR_COLOR, (int(end_x), int(end_y)), 3)

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
            if event.key == pygame.K_c or event.key == pygame.K_r:
                self.add_random_obstacle()
            elif event.key == pygame.K_ESCAPE:
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


def runner(dynamics: Dynamics, lidar_distance: float, lidar_num: int, u_max: float, render: bool, num_steps: int = 1000,
           results_path: Path = None, world_file: Path = None):

    sim_env = SimulationEnv(
        dynamics=dynamics,
        render=render,  # Set to False for headless mode
        border_margin=50,  # Margin for inner boundary
        num_lidar=lidar_num,  # Number of lidar beams
        lidar_distance=lidar_distance,  # Maximum lidar distance
        initial_obstacles=0, 
        u_max=u_max,
        world_file=world_file
    )

    observation, _, = sim_env.reset()

    observation_track = np.zeros((num_steps, observation.shape[0]))
    u_ref_track = np.zeros((num_steps, 2))
    u_actual_track = np.zeros((num_steps, 2))


    for i in range(num_steps):
        # Fetch Pygame events
        for event in pygame.event.get():
            sim_env.process_event(event)

        # Handle user input for acceleration via dynamics' map_keys_to_actions
        keys = pygame.key.get_pressed()
        u = dynamics.map_keys_to_actions(keys)

        # Step the environment
        observation, _, done, info = sim_env.step(u, observation)

        u_used = dynamics.get_used_u()
        sim_env.add_single_frame_non_physics_object({'type': 'arrow', 'start': observation[:2], 'end': observation[:2] + u[:2], 'color': (255, 120, 120)})
        sim_env.add_single_frame_non_physics_object({'type': 'arrow', 'start': observation[:2], 'end': observation[:2] + u_used, 'color': (50, 205, 50)})

        sim_env.render()

        sim_env.set_bars(*(abs(u_used[0]), abs(u_used[1])))

        observation_track[i] = observation
        u_ref_track[i] = u
        u_actual_track[i] = u_used

    # Do plotting and stuff

    if results_path is not None:

        # Create directory if it doesn't exist
        results_path.mkdir(parents=True, exist_ok=True)

        fig_list = []

        for i in range(0, num_steps, 10):
            pos = observation_track[i][:2]
            fig_list.append({"type": "circle", "pos": pos, "radius": 2, "color": (138,43,226)})

        # Create drawings in pygame
        #for i in range(0, num_steps, 50):
        #    start = observation_track[i][:2]
            #fig_list.append({"type": "arrow", "start": start, "end": start + u_ref_track[i][:2], "color": (255, 120, 120)})
        #    fig_list.append({"type": "arrow", "start": start, "end": start + u_actual_track[i][:2], "color": (50, 205, 50)})

        #fig_list.append({"type": "arrow", "start": [600, 500], "end": [600, 500] + u_ref_track[0][:2], "color": (255, 120, 120)})

        sim_env.render_final_figures(fig_list, results_path / "track.png")

        # Calculate u mean squared error, and save to json
        u_mse = np.mean((u_ref_track - u_actual_track) ** 2)
        with open(results_path / "results.json", "w") as f:
            json.dump({"u_mse": u_mse}, f)

        # Create plots of u_ref and u_actual
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))

        ax[0].plot(u_ref_track[:, 0], label="u_ref")
        ax[0].plot(u_actual_track[:, 0], label="u_actual")
        ax[0].set_title("u_x")
        ax[0].legend()
        ax[0].grid()
        ax[0].set_xlabel("Time Steps")
        ax[0].set_ylabel("Acceleration")

        ax[1].plot(u_ref_track[:, 1], label="u_ref")
        ax[1].plot(u_actual_track[:, 1], label="u_actual")
        ax[1].set_title("u_y")
        ax[1].legend()
        ax[1].grid()
        ax[1].set_xlabel("Time Steps")
        ax[1].set_ylabel("Acceleration")

        plt.tight_layout()
        fig.savefig(results_path / "u_plots.pdf")
        plt.close()

        # Create plot for u size
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        u_ref_size = np.linalg.norm(u_ref_track, axis=1)
        u_actual_size = np.linalg.norm(u_actual_track, axis=1)

        ax.plot(u_ref_size, label="u_ref")
        ax.plot(u_actual_size, label="u_actual")
        ax.set_title("u Size")
        ax.legend()
        ax.grid()
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Acceleration")

        plt.tight_layout()
        fig.savefig(results_path / "u_size_plot.pdf")
        plt.close()

        # Create plots for velocity
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))

        ax[0].plot(observation_track[:, 3], label="v_x")
        ax[0].set_title("Velocity x")
        ax[0].legend()
        ax[0].grid()
        ax[0].set_xlabel("Time Steps")
        ax[0].set_ylabel("Velocity")

        ax[1].plot(observation_track[:, 4], label="v_y")
        ax[1].set_title("Velocity y")
        ax[1].legend()
        ax[1].grid()
        ax[1].set_xlabel("Time Steps")
        ax[1].set_ylabel("Velocity")

        plt.tight_layout()
        fig.savefig(results_path / "velocity_plots.pdf")
        plt.close()

        # Plot for velocity size
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        velocity_size = np.linalg.norm(observation_track[:, 3:5], axis=1)

        ax.plot(velocity_size, label="Velocity")
        ax.set_title("Velocity Size")
        ax.legend()
        ax.grid()
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Velocity")

        plt.tight_layout()
        fig.savefig(results_path / "velocity_size_plot.pdf")
        plt.close()

        # Create plots for position
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))

        ax[0].plot(observation_track[:, 0], label="x")
        ax[0].set_title("Position x")
        ax[0].legend()
        ax[0].grid()
        ax[0].set_xlabel("Time Steps")
        ax[0].set_ylabel("Position")

        ax[1].plot(observation_track[:, 1], label="y")
        ax[1].set_title("Position y")
        ax[1].legend()
        ax[1].grid()
        ax[1].set_xlabel("Time Steps")
        ax[1].set_ylabel("Position")

        plt.tight_layout()
        fig.savefig(results_path / "position_plots.pdf")
        plt.close()

    sim_env.close()
    


if __name__ == "__main__":

    U_MAX = 50

    # Set dynamics and parameters
    dynamics = DotDynamicsNormal(dt=1e-2, control_size=U_MAX)
    
    # Run the simulation
    runner(
         dynamics=dynamics,
         lidar_distance=130,
         lidar_num=32,
         u_max=U_MAX,
         render=True,
         results_path=Path("results"),
         num_steps=500,
         world_file=Path("worlds/race_track.json")
    )
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import matplotlib.pyplot as plt
from pathlib import Path
import json
import os
import random

from triangles_nav_linear import DotDynamicsNormalTrianglesCBF
from navigationgym import SimulationEnv, Dynamics


class RandomController:

    def __init__(self, u_max: float, repeat_steps_range: tuple):
        """
        U_RANGE: The range from which to sample the control
        hold_steps: The number of steps to perform the same action
        """

        self.u_max = u_max
        self.repeat_steps_range = repeat_steps_range

        self.current_u = None
        self.steps = 0
        self.repeat_steps = 0

    def get_control(self) -> np.ndarray:

        if self.steps == self.repeat_steps:
            self.current_u = np.random.uniform(-self.u_max, self.u_max, 2)
            self.steps = 0
            self.repeat_steps = np.random.randint(*self.repeat_steps_range)

        #print(self.steps, self.repeat_steps)
        self.steps += 1
        return self.current_u


def ablation_core_runner(dynamics: Dynamics, lidar_distance: float, lidar_num: int, render: bool, controller: RandomController, num_steps: int = 1000,
           results_path: Path = None, world_file: Path = None, initial_obstacles: int = 0, j=0) -> tuple[bool, int]:
    
    """
    Return True if we crashed, False otherwise
    """

    sim_env = SimulationEnv(
        dynamics=dynamics,
        render=render,  # Set to False for headless mode
        border_margin=50,  # Margin for inner boundary
        num_lidar=lidar_num,  # Number of lidar beams
        lidar_distance=lidar_distance,  # Maximum lidar distance
        initial_obstacles=initial_obstacles,
        world_file=world_file
    )

    observation, _, = sim_env.reset()

    #observation_track = np.zeros((num_steps, observation.shape[0]))
    #u_ref_track = np.zeros((num_steps, 2))
    #u_actual_track = np.zeros((num_steps, 2))

    crashed = False
    for i in range(num_steps):
        # Fetch Pygame events
        #for event in pygame.event.get():
        #    sim_env.process_event(event)

        # Handle user input for acceleration via dynamics' map_keys_to_actions
        #keys = pygame.key.get_pressed()
        #u = dynamics.map_keys_to_actions(keys)
        u = controller.get_control()

        # Step the environment
        
        observation, _, done, info = sim_env.step(u, observation, j)

        #print(i, info)

        if info["crash"]:
            return True, i, info["crash_distance"]

        u_used = dynamics.get_used_u()

        if render:
            
            sim_env.add_single_frame_non_physics_object({'type': 'arrow', 'start': observation[:2], 'end': observation[:2] + u[:2], 'color': (255, 120, 120)})
            sim_env.add_single_frame_non_physics_object({'type': 'arrow', 'start': observation[:2], 'end': observation[:2] + u_used, 'color': (50, 205, 50)})


            sim_env.render()

            sim_env.set_bars(*(abs(u_used[0]), abs(u_used[1])))

        #observation_track[i] = observation
        #u_ref_track[i] = u
        #u_actual_track[i] = u_used

    return crashed, num_steps, 0

def ablation_runner(NUM_STEPS, DT, LIDAR_NUM, RENDER, INITIAL_OBSTACLES, U_MAX, REPEAT_STEPS_RANGE, ALPHA_RANGE, LIDAR_RANGE, seed) -> tuple[bool, float, float, float]:

    # Random seed
    #seed = uuid.uuid4().int % 2**32  # Generates a unique integer seed

    # Seed the random number generators
    #random.seed(seed)
    #np.random.seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # For the initial state, we choose v=0, and random position
    #x_0, y_0 = np.random.uniform(55, 745), np.random.uniform(55, 545)
    #x_0 = np.random.uniform(100, 600)
    #y_0 = np.random.uniform(200, 400)
    x_0 = 400
    y_0 = 300

    # Start chosing variables
    controller = RandomController(U_MAX, REPEAT_STEPS_RANGE)
    alpha_1 = np.random.uniform(*ALPHA_RANGE)
    alpha_2 = np.random.uniform(*ALPHA_RANGE)
    lidar_distance = np.random.uniform(*LIDAR_RANGE)

    #print("Seed:", seed, "Alpha 1:", alpha_1, "Alpha 2:", alpha_2, "Lidar distance:", lidar_distance)
    
    crashed, num_steps = True, 0
    i = 0
    while crashed and num_steps == 0:

        dynamics_tri = DotDynamicsNormalTrianglesCBF(dt=DT, p1=alpha_1, p2=alpha_2,
                                                initial_state=np.array([x_0, y_0, 0.0, 0.0, 0.0]))
    
        try:
            while crashed and num_steps == 0:
                crashed, num_steps, crash_distance = ablation_core_runner(dynamics=dynamics_tri, lidar_distance=lidar_distance, lidar_num=LIDAR_NUM, render=RENDER, controller=controller, 
                                num_steps=NUM_STEPS, initial_obstacles=INITIAL_OBSTACLES, j=i)
                i += 1
                print("Retries:", i, "Steps:", num_steps)

        except np.linalg.LinAlgError:
            print("LinAlgError")
            crashed, num_steps = True, 0
        
    return crashed, alpha_1, alpha_2, lidar_distance, crash_distance

def read_and_plot_results(results_dir: Path):

    # Read the json file
    with open(results_dir / "results.json", "r") as f:
        results = json.load(f)

    # Plot the results
    results = np.array(results)

    # Create a 2D plot with alpha_1 and alpha_2 along the axes, and scatter crashes
    fig, ax = plt.subplots()
    #ax.scatter(results[:, 1], results[:, 2], c=results[:, 0], cmap='coolwarm')
    # Plot as red if crashed, green if not
    ax.scatter(results[results[:, 0] == 1, 1], results[results[:, 0] == 1, 2], c='red', label="Crashed")
    ax.scatter(results[results[:, 0] == 0, 1], results[results[:, 0] == 0, 2], c='green', label="Not crashed")

    ax.set_xlabel("Alpha 1")
    ax.set_ylabel("Alpha 2")
    
    # Save as pdf
    fig.savefig(results_dir / "scatter.pdf")

    # Also make a heatmap of failures
    fig, ax = plt.subplots()
    ax.hist2d(results[results[:, 0] == 1, 1], results[results[:, 0] == 1, 2], bins=20, cmap='coolwarm')
    ax.set_xlabel("Alpha 1")
    ax.set_ylabel("Alpha 2")
    fig.colorbar(ax.pcolormesh)
    fig.savefig(results_dir / "hist2d_fails.pdf")



    #plt.show()

def init_pool_processes():
    # Seed NumPy's random number generator
    np.random.seed()  # Seeds with a random seed from OS entropy
    
    # Seed Python's built-in random module
    random.seed()     # Similarly seeds with a random seed from OS entropy
    
    # Optionally, print the PID to verify separate processes
    pid = os.getpid()
    print(f"Process {pid} initialized with unique seeds.")

if __name__ == "__main__":

    # Constants
    RENDER = False
    NUM_STEPS = 4000
    DT = 1e-2
    LIDAR_NUM = 64
    INITIAL_OBSTACLES = 30

    # Random variables intervals
    U_MAX = 200
    REPEAT_STEPS_RANGE = (5, 500)
    ALPHA_RANGE = (0.01, 15)
    LIDAR_RANGE = (10, 800)

    # Prepare the list of arguments for each process
    #crashed, alpha_1, alpha_2, lidar_distance = ablation_runner(NUM_STEPS, DT, LIDAR_NUM, RENDER, INITIAL_OBSTACLES, U_MAX, REPEAT_STEPS_RANGE, ALPHA_RANGE, LIDAR_RANGE)
    #print(crashed)
    args = [
        (NUM_STEPS, DT, LIDAR_NUM, RENDER, INITIAL_OBSTACLES, U_MAX, REPEAT_STEPS_RANGE, ALPHA_RANGE, LIDAR_RANGE, random.randint(0, 2**32))
        for i in range(3)
    ]

    # Run many simulations using multiprocessing with starmap
    with Pool(processes=12, initializer=init_pool_processes) as p:
        results = p.starmap(ablation_runner, args)

    # Save results to a file
    results_dir = Path("ablation_results")
    results_dir.mkdir(exist_ok=True)

    # Dump json
    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f)

    read_and_plot_results(results_dir)
    


    

    
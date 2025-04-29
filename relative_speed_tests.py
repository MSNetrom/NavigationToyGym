from pathlib import Path
import numpy as np
from cbf_tools import FirstOrderGeneralLie, SoftMinLie, single_exponential_cbf_solver
from navigationgym import DotDynamicsNormal, ConstantSpeedObstacle, runner

class Order2RelativeSpeedCBF(FirstOrderGeneralLie):

    def __init__(self, radius: float, u_max: float, p1: float, lidar_num: int):
        self.radius = radius
        self.u_max = u_max
        self.p1 = p1
        self.lidar_num = lidar_num

    def get_Lg_Lf_and_psi(self, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        epsilon = 0
        #pos_vector = 0
        #vel_vector = observation[3:5]

        r = observation[5:5+2*self.lidar_num].reshape(-1, 2)
        r_dot = observation[-2*self.lidar_num:].reshape(-1, 2)
        #r_dot = -observation[3:5].reshape(-1, 2)

        #print((pos_vector - circle_centers).shape)
        h = np.sum(r ** 2, axis=1) - self.radius ** 2 - epsilon
        h_dot = 2*np.sum(r * r_dot, axis=1)

        psi = h_dot + self.p1 * h
        Lf_psi = 2*np.sum(r_dot * r_dot, axis=1) + 2*self.p1*np.sum(r * r_dot, axis=1)

        Lg_psi = - 2 * r

        return Lg_psi, Lf_psi, psi
    

class Order2NonRelativeSpeedCBF(FirstOrderGeneralLie):

    def __init__(self, radius: float, u_max: float, p1: float, lidar_num: int):
        self.radius = radius
        self.u_max = u_max
        self.p1 = p1
        self.lidar_num = lidar_num

    def get_Lg_Lf_and_psi(self, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        epsilon = 0
        #pos_vector = 0
        #vel_vector = observation[3:5]

        r = observation[5:5+2*self.lidar_num].reshape(-1, 2)
        #r_dot = observation[-2*self.lidar_num:].reshape(-1, 2)
        r_dot = -observation[3:5].reshape(-1, 2)

        #print((pos_vector - circle_centers).shape)
        h = np.sum(r ** 2, axis=1) - self.radius ** 2 - epsilon
        h_dot = 2*np.sum(r * r_dot, axis=1)

        psi = h_dot + self.p1 * h
        Lf_psi = 2*np.sum(r_dot * r_dot, axis=1) + 2*self.p1*np.sum(r * r_dot, axis=1)

        Lg_psi = - 2 * r

        return Lg_psi, Lf_psi, psi
    
class DotDynamicsSpeedSoftMin(DotDynamicsNormal):

    def __init__(self, dt: float, radius: float, u_max: float, p1: float, p2: float, 
                 lidar_num: int,
                 initial_state: np.ndarray = np.array([80.0, 80.0, 0.0, 0.0, 0.0]),
                    constant_control: np.ndarray = None, k: float = 5):
        super().__init__(dt=dt, initial_state=initial_state, constant_control=constant_control)
        self.soft_min_cbf = SoftMinLie([Order2RelativeSpeedCBF(radius, u_max, p1, lidar_num)], k=k)
        self.radius = radius
        self.u_max = u_max
        self.p2 = p2

    def perform_step(self, u: np.ndarray, observation: np.ndarray) -> np.ndarray:

        # Get constraints
        Lg_psi, Lf_psi, psi = self.soft_min_cbf.get_Lg_Lf_and_psi(observation)

        # Calculate optimal control
        u_safe = single_exponential_cbf_solver(u_ref=u, Lg_psi=Lg_psi, Lf_psi=Lf_psi, psi=psi, p1=self.p2)

        return super().perform_step(u_safe, observation)
    

class DotDynamicsNonSpeedSoftMin(DotDynamicsNormal):

    def __init__(self, dt: float, radius: float, u_max: float, p1: float, p2: float, 
                 lidar_num: int,
                 initial_state: np.ndarray = np.array([80.0, 80.0, 0.0, 0.0, 0.0]),
                    constant_control: np.ndarray = None, k: float = 5):
        super().__init__(dt=dt, initial_state=initial_state, constant_control=constant_control)
        self.soft_min_cbf = SoftMinLie([Order2NonRelativeSpeedCBF(radius, u_max, p1, lidar_num)], k=k)
        self.radius = radius
        self.u_max = u_max
        self.p2 = p2

    def perform_step(self, u: np.ndarray, observation: np.ndarray) -> np.ndarray:

        # Get constraints
        Lg_psi, Lf_psi, psi = self.soft_min_cbf.get_Lg_Lf_and_psi(observation)

        # Calculate optimal control
        u_safe = single_exponential_cbf_solver(u_ref=u, Lg_psi=Lg_psi, Lf_psi=Lf_psi, psi=psi, p1=self.p2)

        return super().perform_step(u_safe, observation)
    

if __name__ == "__main__":

    U_MAX = 50
    LIDAR_BEAMS = 64
    DT = 1e-2

    # Example: Using your soft-min CBF controlled vehicle dynamics (from soft_min_nav_linear.py)

    dynamics = DotDynamicsSpeedSoftMin(dt=DT, radius=1, u_max=U_MAX, p1=3, p2=2.01, k=0.1,
                                        lidar_num=LIDAR_BEAMS, constant_control=np.array([0.0, 0.0]),
                                        initial_state=np.array([400.0, 300.0, 0.0, 0.0, 0.0]))
    
    dynamic_obs1 = ConstantSpeedObstacle(
         initial_state=np.array([500.0, 300.0, -50.0, 0.0]),  # Speed set to zero = static
         dt=1e-2,
         radius=15,
         color=(0, 0, 255)
    )

    # Run simulation with only dynamic obstacles.
    runner(
         dynamics=dynamics,
         lidar_distance=130,
         lidar_num=LIDAR_BEAMS,
         u_max=U_MAX,
         render=True,
         results_path=Path("results/relative_speed"),
         num_steps=500,
         dynamic_obstacles=[dynamic_obs1],
         dt=DT
    )
    
    dynamics = DotDynamicsNonSpeedSoftMin(dt=DT, radius=1, u_max=U_MAX, p1=3, p2=2.01, k=0.1,
                                        lidar_num=LIDAR_BEAMS, constant_control=np.array([0.0, 0.0]),
                                        initial_state=np.array([400.0, 300.0, 0.0, 0.0, 0.0]))
    
    dynamic_obs1 = ConstantSpeedObstacle(
         initial_state=np.array([500.0, 300.0, -50.0, 0.0]),  # Speed set to zero = static
         dt=1e-2,
         radius=15,
         color=(0, 0, 255)
    )

    # Run simulation with only dynamic obstacles.
    runner(
         dynamics=dynamics,
         lidar_distance=130,
         lidar_num=LIDAR_BEAMS,
         u_max=U_MAX,
         render=True,
         results_path=Path("results/non_relative_speed"),
         num_steps=500,
         dynamic_obstacles=[dynamic_obs1],
         dt=DT
    )


import numpy as np
from pathlib import Path

from cbf_tools import FirstOrderGeneralLie, SoftMinLie, single_exponential_cbf_solver
from navigationgym import DotDynamicsNormal, runner

class MultiObjectCBF2Order(FirstOrderGeneralLie):

    def __init__(self, radius: float, u_max: float, p1: float):
        self.radius = radius
        self.u_max = u_max
        self.p1 = p1
    
    def get_Lg_Lf_and_psi(self, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        epsilon = 0
        pos_vector = 0
        vel_vector = observation[3:5]
        circle_centers = observation[5:].reshape(-1, 2)

        #print((pos_vector - circle_centers).shape)
        h = np.sum((pos_vector - circle_centers) ** 2, axis=1) - self.radius ** 2 - epsilon
        h_dot = 2*np.sum((pos_vector - circle_centers) * vel_vector, axis=1)

        psi = h_dot + self.p1 * h
        Lf_psi = 2*np.dot(vel_vector, vel_vector) + self.p1*h_dot
        Lg_psi = 2*(pos_vector - circle_centers)

        return Lg_psi, Lf_psi, psi
    

class DotDynamicsNormalSoftMin(DotDynamicsNormal):

    def __init__(self, dt: float, radius: float, u_max: float, p1: float, p2: float, initial_state: np.ndarray = np.array([80.0, 80.0, 0.0, 0.0, 0.0]),
                    constant_control: np.ndarray = None, k: float = 5):
        super().__init__(dt=dt, initial_state=initial_state, constant_control=constant_control)
        self.soft_min_cbf = SoftMinLie([MultiObjectCBF2Order(radius, u_max, p1)], k=k)
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
     
    dynamics = DotDynamicsNormalSoftMin(dt=1e-2, radius=1.5, u_max=U_MAX, p1=3, p2=2, k=2)

    # Run the simulation
    runner(
         dynamics=dynamics,
         lidar_distance=100,
         lidar_num=32,
         u_max=U_MAX,
         render=True,
         world_file=Path("worlds/random_track_sparse.json"),
         num_steps=10000,
    )
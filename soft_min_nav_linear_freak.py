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
        h = np.sum((circle_centers - pos_vector) ** 2, axis=1) - self.radius ** 2 - epsilon
        h_dot = 2*np.sum((circle_centers - pos_vector) * (-vel_vector), axis=1)

        # Get the angle of the velocity vector, and the norm
        vel_angle = np.arctan2(vel_vector[1], vel_vector[0])
        vel_norm = np.linalg.norm(vel_vector)

        print("vel_angle: ", vel_angle)

        # Create A matrix
        A = np.array([[np.cos(vel_angle), -vel_norm*np.sin(vel_angle)],
                      [np.sin(vel_angle), vel_norm*np.cos(vel_angle)]])
        
        psi = h_dot + self.p1 * h
        Lf_psi = 2*np.dot(vel_vector, vel_vector) + self.p1*h_dot
        Lg_psi = - 2*(circle_centers - pos_vector) @ A

        print("h min: ", np.min(h))
        print("psi min: ", np.min(psi))

        return Lg_psi, Lf_psi, psi
    

class DotDynamicsNormalSoftMin(DotDynamicsNormal):

    def __init__(self, dt: float, radius: float, u_max: float, p1: float, p2: float, initial_state: np.ndarray = np.array([200.0, 200.0, 0.0, 0.0, 0.0]),
                    constant_control: np.ndarray = None, k: float = 5):
        super().__init__(dt=dt, initial_state=initial_state, constant_control=constant_control)
        self.soft_min_cbf = SoftMinLie([MultiObjectCBF2Order(radius, u_max, p1)], k=k)
        self.radius = radius
        self.u_max = u_max
        self.p2 = p2

    def perform_step(self, u: np.ndarray, observation: np.ndarray) -> np.ndarray:

        vel_vector = observation[3:5]

        # Get the angle of the velocity vector, and the norm
        vel_angle = np.arctan2(vel_vector[1], vel_vector[0])
        vel_norm = np.linalg.norm(vel_vector)

        A = np.array([[np.cos(vel_angle), -vel_norm*np.sin(vel_angle)],
                      [np.sin(vel_angle), vel_norm*np.cos(vel_angle)]])

        # Get constraints
        Lg_psi, Lf_psi, psi = self.soft_min_cbf.get_Lg_Lf_and_psi(observation)

        # Calculate optimal control
        u_safe_polar = single_exponential_cbf_solver(u_ref=u, Lg_psi=Lg_psi, Lf_psi=Lf_psi, psi=psi, p1=self.p2)

        # Convert to cartesian coordinates
        u_safe = A @ u_safe_polar
        

        return super().perform_step(u_safe, observation)
    
if __name__ == "__main__":
    
    U_MAX = 50
     
    dynamics = DotDynamicsNormalSoftMin(dt=1e-2, radius=10, u_max=U_MAX, p1=3, p2=1, k=1)

    # Run the simulation
    runner(
         dynamics=dynamics,
         lidar_distance=130,
         lidar_num=64,
         u_max=U_MAX,
         render=True,
         initial_obstacles=10,
         #world_file=Path('worlds/tight_track.json'),
         num_steps=100000,
    )
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

        epsilon = 1

        vel_vector = observation[3:5]
        lidar_vecs = observation[5:].reshape(-1, 2)
        lidar_vecs_dot = np.zeros_like(lidar_vecs) - vel_vector

        # Create variables for later use
        # Shift lidar vecs, lidar_vec[i] = lidar_vec[i-1]
        lidars_shifted = np.zeros_like(lidar_vecs)
        lidars_shifted[:-1] = lidar_vecs[1:]
        lidars_shifted[-1] = lidar_vecs[0]

        lidars_shifted_dot = np.zeros_like(lidar_vecs_dot)
        lidars_shifted_dot[:-1] = lidar_vecs_dot[1:]
        lidars_shifted_dot[-1] = lidar_vecs_dot[0]

        # Create usefull vectors
        lidar_vecs_norm = np.linalg.norm(lidar_vecs, axis=1)
        lidars_shifted_norm = np.linalg.norm(lidars_shifted, axis=1)
        
        # Compute the h and stuff
        h = np.einsum('ij,ij->i', lidar_vecs, lidars_shifted) + lidar_vecs_norm * lidars_shifted_norm - epsilon
        h_dot = np.einsum('ij,ij->i', lidar_vecs_dot, lidars_shifted) + np.einsum('ij,ij->i', lidar_vecs, lidars_shifted_dot) + np.einsum('ij,ij->i', lidar_vecs, lidar_vecs_dot) * lidars_shifted_norm / lidar_vecs_norm + np.einsum('ij,ij->i', lidars_shifted, lidars_shifted_dot) * lidar_vecs_norm / lidars_shifted_norm
        

        psi = h_dot + self.p1 * h

        Lg_psi = - lidar_vecs - lidars_shifted - lidar_vecs_norm[:, np.newaxis] * lidars_shifted / lidars_shifted_norm[:, np.newaxis] - lidars_shifted_norm[:, np.newaxis] * lidar_vecs / lidar_vecs_norm[:, np.newaxis]
        #Lf_psi = 2 * np.einsum('ij,ij->i', lidar_vecs_dot, lidars_shifted_dot) - np.einsum('ij,ij->i', lidar_vecs_dot, lidar_vecs) ** 2 * lidars_shifted_norm / lidar_vecs_norm ** 3 + 2 * np.einsum('ij,ij->i', lidar_vecs_dot, lidar_vecs_dot) * lidars_shifted_norm / lidar_vecs_norm + 2 * np.einsum('ij,ij->i', lidar_vecs, lidar_vecs_dot) * np.einsum('ij,ij->i', lidars_shifted, lidars_shifted_dot) / (lidar_vecs_norm * lidars_shifted_norm) - np.einsum('ij,ij->i', lidars_shifted, lidars_shifted_dot) ** 2 * lidar_vecs_norm / lidar_vecs_norm ** 3 + np.einsum('ij,ij->i', lidars_shifted_dot, lidars_shifted_dot) * lidar_vecs_norm / lidar_vecs_norm + self.p1 * h_dot

        # Corrected L_fψ calculation
        term1 = 2 * np.einsum('ij,ij->i', lidar_vecs_dot, lidars_shifted_dot)
        term2 = - (np.einsum('ij,ij->i', lidar_vecs, lidar_vecs_dot)**2 * lidars_shifted_norm) / (lidar_vecs_norm**3)
        term3 = (np.einsum('ij,ij->i', lidar_vecs_dot, lidar_vecs_dot) * lidars_shifted_norm) / lidar_vecs_norm  # Removed factor of 2
        term4 = 2 * (np.einsum('ij,ij->i', lidar_vecs, lidar_vecs_dot) * np.einsum('ij,ij->i', lidars_shifted, lidars_shifted_dot)) / (lidar_vecs_norm * lidars_shifted_norm)
        term5 = - (np.einsum('ij,ij->i', lidars_shifted, lidars_shifted_dot)**2 * lidar_vecs_norm) / (lidars_shifted_norm**3)  # Fixed denominator
        term6 = (np.einsum('ij,ij->i', lidars_shifted_dot, lidars_shifted_dot) * lidar_vecs_norm) / lidars_shifted_norm  # Added division
        Lf_psi = term1 + term2 + term3 + term4 + term5 + term6 + self.p1 * h_dot

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

        vel_vector = observation[3:5]

        # Get constraints
        Lg_psi, Lf_psi, psi = self.soft_min_cbf.get_Lg_Lf_and_psi(observation)

        # Calculate optimal control
        u_safe = single_exponential_cbf_solver(u_ref=u, Lg_psi=Lg_psi, Lf_psi=Lf_psi, psi=psi, p1=self.p2)

        # Max length of u_safe is 50
        #u_safe = np.clip(u_safe, -self.u_max, self.u_max)
        return super().perform_step(u_safe, observation)

if __name__ == "__main__":
    
    U_MAX = 50
     
    dynamics = DotDynamicsNormalSoftMin(dt=1e-2, radius=10, u_max=U_MAX, p1=4, p2=3.01, k=1) #p1=3, p2=2, k=1)

    # Run the simulation
    runner(
         dynamics=dynamics,
         lidar_distance=130,
         lidar_num=64,
         u_max=U_MAX,
         render=True,
         #initial_obstacles=20,
         world_file=Path('worlds/random_track_quadrant.json'),
         num_steps=100000,
    )
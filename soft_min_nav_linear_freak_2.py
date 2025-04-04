import numpy as np
from pathlib import Path

from cbf_tools import FirstOrderGeneralLie, SoftMinLie, single_exponential_cbf_solver
from navigationgym import DotDynamicsNormal, runner

def angle_between_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute the angle (in radians) between corresponding 2D vectors in a and b.
    
    Parameters:
        a (np.ndarray): An array of shape (N, 2), where each row is a 2D vector.
        b (np.ndarray): An array of shape (N, 2), where each row is a 2D vector.

    Returns:
        np.ndarray: A 1D array of shape (N,) containing the angles in radians
                    between the corresponding rows of a and b.
    """
    # Compute dot product for each corresponding pair.
    dot = np.einsum('ij,ij->i', a, b)
    # Compute the magnitude of the 2D "cross product".
    cross = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    # Compute the angle using arctan2.
    return np.arctan2(cross, dot)

class MultiObjectCBF2Order(FirstOrderGeneralLie):

    def __init__(self, radius: float, u_max: float, p1: float):
        self.radius = radius
        self.u_max = u_max
        self.p1 = p1
    
    def get_Lg_Lf_and_psi(self, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        epsilon = 5

        vel_vector = observation[3:5]
        lidar_vecs = observation[5:].reshape(-1, 2)
        print(lidar_vecs)
        lidar_vecs_dot = np.zeros_like(lidar_vecs) - vel_vector

        # Create variables for later use
        # Shift lidar vecs, lidar_vec[i] = lidar_vec[i-1]
        lidars_shifted = np.zeros_like(lidar_vecs)
        lidars_shifted[:-1] = lidar_vecs[1:]
        lidars_shifted[-1] = lidar_vecs[0]

        lidars_shifted_dot = np.zeros_like(lidar_vecs_dot)
        lidars_shifted_dot[:-1] = lidar_vecs_dot[1:]
        lidars_shifted_dot[-1] = lidar_vecs_dot[0]

        # Create triangular constraints from lidar beams (N, 2)
        l_1 = lidar_vecs
        s = lidars_shifted - lidar_vecs
        s_dot = lidars_shifted_dot - lidar_vecs_dot
        l_2 = lidars_shifted

        normal_transformer = np.array([[0, -1], [1, 0]])
        n = s @ normal_transformer.T
        n_dot = s_dot @ normal_transformer.T
        n_norm = np.linalg.norm(n, axis=1)


        d = n / n_norm[:, np.newaxis]
        d_dot = n_dot / n_norm[:, np.newaxis] - np.einsum('ij,ij->i', n, n_dot)[:, np.newaxis] * n / n_norm[:, np.newaxis]**3
        d_dot_dot = - 2 * np.einsum('ij,ij->i', n, n_dot)[:, np.newaxis] * n_dot / n_norm[:, np.newaxis]**3 - np.einsum('ij,ij->i', n_dot, n_dot)[:, np.newaxis] * n / n_norm[:, np.newaxis]**3 + 3 * np.einsum('ij,ij->i', n, n_dot)[:, np.newaxis] ** 2 * n / n_norm[:, np.newaxis]**5

        # Find the angles between l1 s and l2 s
        l1_s_angle = angle_between_vectors(lidar_vecs, s)
        l2_s_angle = angle_between_vectors(lidars_shifted, s)

        # Calculate the possible Lg_psi and Lf_psi
        lidar_vecs_norm = np.linalg.norm(lidar_vecs, axis=1)
        h_0 = lidar_vecs_norm - epsilon
        h_0_dot = np.einsum('ij,ij->i', lidar_vecs, lidar_vecs_dot) / lidar_vecs_norm

        psi_0 = h_0_dot + self.p1 * h_0
        Lg_psi_0 = - lidar_vecs / lidar_vecs_norm[:, np.newaxis]
        Lf_psi_0 = np.einsum('ij,ij->i', lidar_vecs_dot, lidar_vecs_dot) / lidar_vecs_norm - np.einsum('ij,ij->i', lidar_vecs, lidar_vecs_dot) ** 2 / lidar_vecs_norm**3 + self.p1 * h_0_dot

        h_1 = - np.einsum('ij,ij->i', d, lidar_vecs) - epsilon
        h_1_dot = - np.einsum('ij,ij->i', d_dot, lidar_vecs) - np.einsum('ij,ij->i', d, lidar_vecs_dot)

        psi_1 = h_1_dot + self.p1 * h_1
        Lg_psi_1 = d
        Lf_psi_1 = - np.einsum('ij,ij->i', d_dot_dot, lidar_vecs) - 2*np.einsum('ij,ij->i', d_dot, lidar_vecs_dot) + self.p1 * h_1_dot

        # Print minimum h_0 and h_1

        # If the angle between l1 and s is more than 90 degrees and bigger than the angle between l2 and s, then use psi_0(l_i) and Lg_psi_0(l_i), Lf_psi_0(l_i)
        # If the angle between l2 and s is more than 90 degrees and bigger than the angle between l1 and s, then use psi_0(l_i+1) and Lg_psi_0(l_i+1), Lf_psi_0(l_i+1)
        # Otherwise, use psi_1(l_i) and Lg_psi_1(l_i), Lf_psi_1(l_i)

        psi_0_shifted = np.zeros_like(psi_0)
        psi_0_shifted[:-1] = psi_0[1:]
        psi_0_shifted[-1] = psi_0[0]

        Lg_psi_0_shifted = np.zeros_like(Lg_psi_0)
        Lg_psi_0_shifted[:-1] = Lg_psi_0[1:]
        Lg_psi_0_shifted[-1] = Lg_psi_0[0]

        Lf_psi_0_shifted = np.zeros_like(Lf_psi_0)
        Lf_psi_0_shifted[:-1] = Lf_psi_0[1:]
        Lf_psi_0_shifted[-1] = Lf_psi_0[0]

        l1_s_bigger_than_90 = l1_s_angle > np.pi/2
        l2_s_bigger_than_90 = l2_s_angle > np.pi/2

        l1_bigger_than_l2 = l1_s_angle > l2_s_angle

        psi = psi_0 * (l1_s_bigger_than_90 & l1_bigger_than_l2) + psi_0_shifted * (l2_s_bigger_than_90 & ~l1_bigger_than_l2) + psi_1 * (~l1_s_bigger_than_90 & ~l2_s_bigger_than_90)
        Lg_psi = Lg_psi_0 * (l1_s_bigger_than_90 & l1_bigger_than_l2)[:, np.newaxis] + Lg_psi_0_shifted * (l2_s_bigger_than_90 & ~l1_bigger_than_l2)[:, np.newaxis] + Lg_psi_1 * (~l1_s_bigger_than_90 & ~l2_s_bigger_than_90)[:, np.newaxis]
        Lf_psi = Lf_psi_0 * (l1_s_bigger_than_90 & l1_bigger_than_l2) + Lf_psi_0_shifted * (l2_s_bigger_than_90 & ~l1_bigger_than_l2) + Lf_psi_1 * (~l1_s_bigger_than_90 & ~l2_s_bigger_than_90)

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

        # Get constraints
        Lg_psi, Lf_psi, psi = self.soft_min_cbf.get_Lg_Lf_and_psi(observation)

        # Calculate optimal control
        u_safe = single_exponential_cbf_solver(u_ref=u, Lg_psi=Lg_psi, Lf_psi=Lf_psi, psi=psi, p1=self.p2)

        # Max length of u_safe is 50
        #u_safe = np.clip(u_safe, -self.u_max, self.u_max)
        return super().perform_step(u_safe, observation)

if __name__ == "__main__":
    
    U_MAX = 50
     
    dynamics = DotDynamicsNormalSoftMin(dt=1e-2, radius=10, u_max=U_MAX, p1=3, p2=2, k=1)

    # Run the simulation
    runner(
         dynamics=dynamics,
         lidar_distance=130,
         lidar_num=64,
         u_max=U_MAX,
         render=True,
         initial_obstacles=0,
         #world_file=Path('worlds/tight_track.json'),
         num_steps=100000,
    )
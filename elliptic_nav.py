import numpy as np
from pathlib import Path

from cbf_tools import FirstOrderGeneralLie, SoftMinLie, single_exponential_cbf_solver
from navigationgym import DotDynamicsNormal, runner

def elliptic_contructer(lidar_vecs: np.ndarray, lidar_vec_dots: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    Lambda = np.array([[0.1, 0], [0, 10]])
    k = 1/6
    #Lambda = np.array([[1, 0], [0, 1]])
    #k = 1/2
    alpha_1 = 50
    epsilon = 10

    #lidar_vecs = lidar_vecs - 15*lidar_vecs/np.linalg.norm(lidar_vecs, axis=1)[:, np.newaxis]
    #print(lidar_vecs)

    # Move all tops backwards in the direction of the normal vector between them,
    # And such that they still intersect as they should

    # Get index of the vector in which the direction

    # Shift lidar vecs, lidar_vec[i] = lidar_vec[i-1]
    lidars_shifted = np.zeros_like(lidar_vecs)
    lidars_shifted[:-1] = lidar_vecs[1:]
    lidars_shifted[-1] = lidar_vecs[0]

    # Shift lidar_vec_dots, lidar_vec_dots[i] = lidar_vec_dots[i-1]
    lidar_vec_dots_shifted = np.zeros_like(lidar_vec_dots)
    lidar_vec_dots_shifted[:-1] = lidar_vec_dots[1:]
    lidar_vec_dots_shifted[-1] = lidar_vec_dots[0]

    # Create s? Correct sign / direction? (N, 2)
    s = lidars_shifted - lidar_vecs
    s_dots = lidar_vec_dots_shifted - lidar_vec_dots

    print("S1:", s[0])

    # Create r
    r = (lidars_shifted + lidar_vecs) / 2
    r_dots = (lidar_vec_dots_shifted + lidar_vec_dots) / 2

    # Vectorized approach to construct R_hat without a loop. This has shape (N, 2, 2)
    R_hat = np.empty((s.shape[0], 2, 2))
    R_hat[:, 0, 0] = s[:, 0]
    R_hat[:, 0, 1] = -s[:, 1]
    R_hat[:, 1, 0] = s[:, 1]
    R_hat[:, 1, 1] = s[:, 0]

    R_hat_dots = np.empty((s_dots.shape[0], 2, 2))
    R_hat_dots[:, 0, 0] = s_dots[:, 0]
    R_hat_dots[:, 0, 1] = -s_dots[:, 1]
    R_hat_dots[:, 1, 0] = s_dots[:, 1]
    R_hat_dots[:, 1, 1] = s_dots[:, 0]

    # Calculate some P matrices for later use. They will have shape (N, 2, 2)
    P_1 = R_hat @ Lambda @ R_hat.transpose(0, 2, 1)
    P_2 = R_hat_dots @ Lambda @ R_hat.transpose(0, 2, 1)

    # More calculations
    h = np.einsum('ni,nij,nj->n', r, P_1, r) - k**2 * np.einsum('ni,ni->n', s, s)**2 - epsilon

    # Print minimum h
    print("Minimum h:", np.min(h))

    psi_1 = 2 * np.einsum('ni,nij,nj->n', r_dots, P_1, r) + 2 * np.einsum('ni,nij,nj->n', r, P_2, r) - 4*k**2 * np.einsum('ni,ni,nj,nj->n', s, s, s, s_dots) + alpha_1 * h
    print("Psi1 min:", np.min(psi_1))

    Lg_psi_1 = -2 * np.einsum('ni,nij->nj', r, P_1)
    Lf_psi_1 = 4 * np.einsum('ni,nij,nj->n', r_dots, P_2, r_dots) + 4 * np.einsum('ni,nij,nj->n', r_dots, P_2.transpose(0, 2, 1), r) + 2 * np.einsum('ni,nij,nj->n', r_dots, P_1, r_dots) + 2 * np.einsum('ni,nij,nj->n', r, R_hat_dots @ Lambda @ R_hat_dots.transpose(0, 2, 1), r) - 8 * k**2 * np.einsum('ni,ni,nj,nj->n', s, s_dots, s, s_dots) - 4 * k ** 2 * np.einsum('ni,ni,nj,nj->n', s, s, s_dots, s_dots)

    return Lg_psi_1, Lf_psi_1, psi_1
        

class MultiObjectCBF2OrderElliptic(FirstOrderGeneralLie):

    def __init__(self):
        pass
    
    def get_Lg_Lf_and_psi(self, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        pos_vector = 0
        vel_vector = observation[3:5]
        circle_centers = observation[5:].reshape(-1, 2)

        lidar_vecs = circle_centers - pos_vector
        lidar_vec_dots = np.tile(vel_vector, (lidar_vecs.shape[0], 1))

        Lg_psi_1, Lf_psi_1, psi_1 = elliptic_contructer(lidar_vecs, lidar_vec_dots)

        return Lg_psi_1, Lf_psi_1, psi_1

        """#print((pos_vector - circle_centers).shape)
        h = np.sum((pos_vector - circle_centers) ** 2, axis=1) - self.radius ** 2 - epsilon
        h_dot = 2*np.sum((pos_vector - circle_centers) * vel_vector, axis=1)

        psi = h_dot + self.p1 * h
        Lf_psi = 2*np.dot(vel_vector, vel_vector) + self.p1*h_dot
        Lg_psi = 2*(pos_vector - circle_centers)

        return Lg_psi, Lf_psi, psi"""
    

class DotDynamicsNormalSoftMin(DotDynamicsNormal):

    def __init__(self, dt: float, initial_state: np.ndarray = np.array([80.0, 80.0, 0.0, 0.0, 0.0]),
                    constant_control: np.ndarray = None, k: float = 5):
        super().__init__(dt=dt, initial_state=initial_state, constant_control=constant_control)
        self.soft_min_cbf = SoftMinLie([MultiObjectCBF2OrderElliptic()], k=k)
        self.p2 = 60

    def perform_step(self, u: np.ndarray, observation: np.ndarray) -> np.ndarray:

        # Get constraints
        Lg_psi, Lf_psi, psi = self.soft_min_cbf.get_Lg_Lf_and_psi(observation)

        # Calculate optimal control
        u_safe = single_exponential_cbf_solver(u_ref=u, Lg_psi=Lg_psi, Lf_psi=Lf_psi, psi=psi, p1=self.p2)

        return super().perform_step(u_safe, observation)
    
if __name__ == "__main__":
    
    U_MAX = 50
     
    dynamics = DotDynamicsNormalSoftMin(dt=1e-2, k=1e-3)

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
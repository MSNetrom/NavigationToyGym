import numpy as np
from pathlib import Path

from cbf_tools import FirstOrderGeneralLie, SoftMinLie, single_exponential_cbf_solver
from navigationgym import DotDynamicsNormal, runner

def elliptic_contructer(lidar_vecs: np.ndarray, lidar_vec_dots: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    velocity_vector = np.array([1, 0])

    Lambda = np.array([[0.1, 0], [0, 10]])
    k = 1/6
    #Lambda = np.array([[1, 0], [0, 1]])
   # k = 1/2
    alpha_1 = 5
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
    s_norm = np.linalg.norm(s, axis=1) # Shape (N,)

    # Create triangular constraints from lidar beams (N, 2)
    l_0 = np.zeros_like(lidar_vecs) # Start pos of line 1
    d_0 = lidar_vecs # Direction of line 1
    l_1 = lidar_vecs
    d_1 = lidars_shifted - lidar_vecs
    l_2 = lidars_shifted
    d_2 = - lidars_shifted

    # Find the index in which the velocity vector lies within
    
    # Check the length if lidar_vecs. That is the nu,ber of sectors in existance

    # Get the angle of the velocity vector
    vel_angle = np.arctan2(velocity_vector[1], velocity_vector[0])

    sector_index = np.floor(vel_angle / (2 * np.pi / lidar_vecs.shape[0]))

    # Generate normal vectors
    normal_transformer = np.array([[0, -1],[1, 0]])
    n_0 = d_0 @ normal_transformer.T
    n_1 = d_1 @ normal_transformer.T
    n_2 = d_2 @ normal_transformer.T

    # Compute h, h_dot and Lf_h and Lg_h
    h_0 = np.sum(n_0 * (pos_vector - l_0), axis=1)
    h_1 = np.sum(n_1 * (pos_vector - l_1), axis=1)
    h_2 = np.sum(n_2 * (pos_vector - l_2), axis=1)

    
    h_dot_0 = np.sum(n_0 * vel_vector, axis=1)
    Lf_0 = 0
    h_dot_1 = np.sum(n_1 * vel_vector, axis=1)
    Lf_1 = 0
    h_dot_2 = np.sum(n_2 * vel_vector, axis=1)
    Lf_2 = 0

    # Create alphas and betas for alpha^T u >= beta
    alphas_0 = n_0
    betas_0 = - (Lf_0 + (p1 + p2) * h_dot_0 + p1 * p2 * h_0)
    alphas_1 = n_1
    betas_1 = - (Lf_1 + (p1 + p2) * h_dot_1 + p1 * p2 * h_1)
    alphas_2 = n_2
    betas_2 = - (Lf_2 + (p1 + p2) * h_dot_2 + p1 * p2 * h_2)

    # Evaluate if u_ref itself satisfies any of the triangles from before
    satisfied_0 = alphas_0 @ u_ref >= betas_0
    satisfied_1 = alphas_1 @ u_ref >= betas_1
    satisfied_2 = alphas_2 @ u_ref >= betas_2

    # If u_ref satisfies all 0, 1 and 2 at the same time for some i, then return u_ref
    if np.any(satisfied_0 & satisfied_1 & satisfied_2):
        return u_ref

    alphas_0_lengths_squared = np.sum(alphas_0 ** 2, axis=1)
    alphas_1_lengths_squared = np.sum(alphas_1 ** 2, axis=1)
    alphas_2_lengths_squared = np.sum(alphas_2 ** 2, axis=1)

    # If alphas squared length is less than 1e-6, then set it to NaN
    alphas_0_lengths_squared[alphas_0_lengths_squared < 1e-6] = np.nan
    alphas_1_lengths_squared[alphas_1_lengths_squared < 1e-6] = np.nan
    alphas_2_lengths_squared[alphas_2_lengths_squared < 1e-6] = np.nan

    # Solve the Quad-Prog problem with respect to single active constraints, a^T u = b
    # As shown in my project, when the constraints are active, the solution is u = (b - a^T u_ref) * a / ||a||^2 + u_ref
    #print((betas_0 - alphas_0 @ u_ref).shape, alphas_0.shape)
    u_singles_0 = (betas_0 - alphas_0 @ u_ref)[:, np.newaxis] * alphas_0 / alphas_0_lengths_squared[:, np.newaxis] + u_ref
    u_singles_1 = (betas_1 - alphas_1 @ u_ref)[:, np.newaxis] * alphas_1 / alphas_1_lengths_squared[:, np.newaxis] + u_ref
    u_singles_2 = (betas_2 - alphas_2 @ u_ref)[:, np.newaxis] * alphas_2 / alphas_2_lengths_squared[:, np.newaxis] + u_ref

    # Keep only the ones that satisfy the constraints
    u_singles_0 = u_singles_0[(np.sum(alphas_1 * u_singles_0, axis=1) >= betas_1) & (np.sum(alphas_2 * u_singles_0, axis=1) >= betas_2)]
    u_singles_1 = u_singles_1[(np.sum(alphas_0 * u_singles_1, axis=1) >= betas_0) & (np.sum(alphas_2 * u_singles_1, axis=1) >= betas_2)]
    u_singles_2 = u_singles_2[(np.sum(alphas_0 * u_singles_2, axis=1) >= betas_0) & (np.sum(alphas_1 * u_singles_2, axis=1) >= betas_1)]

    # Solve different combinations of 2 and 2 of these constraints with np linalg solve
    # alphas (N, 2), betas (N,)?
    # Extend alphas from (N, 2) to (N, 1, 2) and concat with other alphas on dimension 1
    # Extend betas from (N,) to (N, 1) and concat with other betas on dimension 1
    alphas_0 = alphas_0.reshape(-1, 1, 2)
    alphas_1 = alphas_1.reshape(-1, 1, 2)
    alphas_2 = alphas_2.reshape(-1, 1, 2)
    betas_0 = betas_0.reshape(-1, 1)
    betas_1 = betas_1.reshape(-1, 1)
    betas_2 = betas_2.reshape(-1, 1)

    alpha_comb_0_1 = np.concatenate([alphas_0, alphas_1], axis=1)
    beta_comb_0_1 = np.concatenate([betas_0, betas_1], axis=1)[:, :, np.newaxis]
    alpha_comb_0_2 = np.concatenate([alphas_0, alphas_2], axis=1)
    beta_comb_0_2 = np.concatenate([betas_0, betas_2], axis=1)[:, :, np.newaxis]
    alpha_comb_1_2 = np.concatenate([alphas_1, alphas_2], axis=1)
    beta_comb_1_2 = np.concatenate([betas_1, betas_2], axis=1)[:, :, np.newaxis]

    # Solve the different combinations
    u_safe_0_1 = np.linalg.solve(alpha_comb_0_1, beta_comb_0_1).squeeze(-1)
    u_safe_0_2 = np.linalg.solve(alpha_comb_0_2, beta_comb_0_2).squeeze(-1)
    u_safe_1_2 = np.linalg.solve(alpha_comb_1_2, beta_comb_1_2).squeeze(-1)

    # Concat all proposals, and choose the one that is closest to the reference
    u_proposals = np.concatenate([u_singles_0, u_singles_1, u_singles_2, u_safe_0_1, u_safe_0_2, u_safe_1_2], axis=0)

    # Choose the u that is closest to the reference
    u_safe = u_proposals[np.nanargmin(np.sum((u_proposals - u_ref) ** 2, axis=1))]

    return u_safe

    return Lg_psi_1, Lf_psi_1, psi_1
        

class MultiObjectCBF2OrderElliptic(FirstOrderGeneralLie):

    def __init__(self):
        pass
    
    def get_Lg_Lf_and_psi(self, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        pos_vector = 0
        vel_vector = observation[3:5]
        circle_centers = observation[5:].reshape(-1, 2)

        lidar_vecs = circle_centers - pos_vector
        lidar_vec_dots = - np.tile(vel_vector, (lidar_vecs.shape[0], 1))

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
        self.p2 = 6

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
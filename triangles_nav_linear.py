import numpy as np
from pathlib import Path
from navigationgym import DotDynamicsNormal, runner

def traingles_solver_2d(u_ref: np.ndarray, states: np.ndarray, lidar_vecs: np.ndarray, p1: float, p2: float) -> np.ndarray:
    # Get pos and vel vector
        pos_vector = 0
        vel_vector = states[3:5]

        # Shift lidar vecs, lidar_vec[i] = lidar_vec[i-1]
        lidars_shifted = np.zeros_like(lidar_vecs)
        lidars_shifted[:-1] = lidar_vecs[1:]
        lidars_shifted[-1] = lidar_vecs[0]

        # Create triangular constraints from lidar beams (N, 2)
        l_0 = np.zeros_like(lidar_vecs) # Start pos of line 1
        d_0 = lidar_vecs # Direction of line 1
        l_1 = lidar_vecs
        d_1 = lidars_shifted - lidar_vecs
        l_2 = lidars_shifted
        d_2 = - lidars_shifted

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

class DotDynamicsNormalTrianglesCBF(DotDynamicsNormal):
      
    def __init__(self, dt: float, p1: float, p2: float, control_size: float, initial_state: np.ndarray, constant_control: np.ndarray = None):
        super().__init__(dt, control_size=control_size, initial_state=initial_state, constant_control=constant_control)
        self.p1 = p1
        self.p2 = p2

    def perform_step(self, u: np.ndarray, observation: np.ndarray) -> np.ndarray:
         
        # Find safe u
        states = observation[:len(self.initial_state)]
        lidar_vecs = observation[len(self.initial_state):].reshape(-1, 2)
        u_safe = traingles_solver_2d(u_ref=u, states=states, lidar_vecs=lidar_vecs, p1=self.p1, p2=self.p2)

        # Perform step
        return super().perform_step(u_safe, observation)
    

if __name__ == "__main__":

    #tra

    U_MAX = 50.0
     
    dynamics = DotDynamicsNormalTrianglesCBF(dt=1e-2, p1=1, p2=2, control_size=U_MAX,
                                             initial_state=np.array([51.0, 51.0, 0.0, 0.0, 0.0]),
                                             constant_control=np.array([U_MAX, U_MAX]))

    # Run the simulation
    runner(
         dynamics=dynamics,
         lidar_distance=130,
         lidar_num=32,
         u_max=U_MAX,
         render=True,
         num_steps=8000,
         results_path=Path("triangle_results"),
         world_file=Path("worlds/random_track_sparse.json")
         #world_file=Path("worlds/random_track_quadrant.json")
         #world_file=Path("worlds/zigzag_tight_track.json")
         #world_file=Path("worlds/tight_track.json")
    )

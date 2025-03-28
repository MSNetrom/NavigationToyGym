import numpy as np
from pathlib import Path

from cbf_tools import FirstOrderGeneralLie, SoftMinLie, single_exponential_cbf_solver
from navigationgym import DotDynamicsNormal, runner

class RotationalCBF2Order(FirstOrderGeneralLie):
    """
    Implementation of the rotational CBF as described in the mathematical derivation:
    h = r^T R Λ R^T r - p^2 > 0
    
    Where:
    - R is a rotation matrix that rotates ||dot_r||e_1 into dot_r
    - Λ is a diagonal eigenvalue matrix with λ_1 > λ_2
    """

    def __init__(self, p: float, lambda1: float, lambda2: float, u_max: float, p1: float):
        """
        Initialize the rotational CBF.
        
        Args:
            p: Radius parameter for the safety region
            lambda1: First eigenvalue (longitudinal direction)
            lambda2: Second eigenvalue (lateral direction), should be larger than lambda1
            u_max: Maximum control input
            p1: CBF parameter
        """
        self.p = p
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.u_max = u_max
        self.p1 = p1
        # Lambda matrix as diagonal matrix of eigenvalues
        self.Lambda = np.array([[lambda1, 0], [0, lambda2]])
        # Regularization constant to avoid numerical issues
        self.epsilon = 1e-6
    
    def get_rotation_matrix(self, vel_vector: np.ndarray) -> np.ndarray:
        """
        Computes the rotation matrix R that rotates ||dot_r||e_1 into dot_r.
        
        Args:
            vel_vector: The velocity vector [dot_r_1, dot_r_2]
            
        Returns:
            2x2 rotation matrix
        """
        # Regularize velocity to avoid numerical issues when velocity is close to zero
        vel_reg = vel_vector + self.epsilon * np.array([1, 0])
        vel_norm = np.linalg.norm(vel_reg)
        
        # Compute rotation matrix components
        R = np.array([
            [vel_reg[0] / vel_norm, -vel_reg[1] / vel_norm],
            [vel_reg[1] / vel_norm, vel_reg[0] / vel_norm]
        ])
        
        return R
    
    def get_Lg_Lf_and_psi(self, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes Lg, Lf, and psi for the CBF.
        
        Args:
            observation: State observation vector
            
        Returns:
            Tuple of (Lg_psi, Lf_psi, psi)
        """
        # Extract position and velocity from observation
        pos_vector = 0  # Origin as the reference point
        vel_vector = observation[3:5]
        circle_centers = observation[5:].reshape(-1, 2)
        
        # For each obstacle, compute the barrier function
        psi_values = []
        Lg_psi_values = []
        Lf_psi_values = []
        
        for center in circle_centers:
            # Relative position vector (r)
            r = pos_vector - center
            
            # Compute rotation matrix R
            R = self.get_rotation_matrix(vel_vector)
            
            # Compute h = r^T R Λ R^T r - p^2
            h = r @ R @ self.Lambda @ R.T @ r - self.p**2
            
            # Compute J (skew-symmetric matrix in 2D)
            J = np.array([[0, -1], [1, 0]])
            
            # Compute angular velocity ω = (r_dot_1 * r_ddot_2 - r_dot_2 * r_ddot_1) / ||r_dot||^2
            # Since r_ddot depends on the control input u, we set this term to zero
            # and it will be handled by Lg_psi
            
            # Compute h_dot = 2*dot_r^T R Λ R^T r
            #h_dot = 2 * vel_vector @ R @ self.Lambda @ R.T @ r
            
            # Compute f(r, dot_r) = 2*dot_r^T R Λ R^T r
            #f_r_dot_r = h_dot
            
            # Compute g(r, dot_r) term
            # This represents the coefficient of u in the CBF derivative
            vel_norm_squared = np.linalg.norm(vel_vector)**2 + self.epsilon
            
            # g(r, dot_r) = [(-dot_r_2 * (r^T J R Λ R^T r)), (dot_r_1 * (r^T J R Λ R^T r))] / ||dot_r||^2
            r_JR_Lambda_RT_r = r @ J @ R @ self.Lambda @ R.T @ r
            g_r_dot_r = 2 * np.array([
                -vel_vector[1] * r_JR_Lambda_RT_r,
                vel_vector[0] * r_JR_Lambda_RT_r
            ]) / vel_norm_squared
            
            # Compute psi = h_dot + p1 * h
            #psi = h_dot + self.p1 * h
            
            # Compute Lf_psi (coefficient of time derivative without control input)
            # Only includes the effect of current velocity, not acceleration
            #Lf_psi = self.p1 * h_dot

            Lf_psi = 2 * vel_vector @ R @ self.Lambda @ R.T @ r
            Lg_psi = g_r_dot_r
            psi = h
            
            # Store values for this obstacle
            psi_values.append(psi)
            Lg_psi_values.append(Lg_psi)
            Lf_psi_values.append(Lf_psi)
        
        # Convert to numpy arrays
        psi_values = np.array(psi_values)
        Lg_psi_values = np.array(Lg_psi_values)
        Lf_psi_values = np.array(Lf_psi_values)
        
        return Lg_psi_values, Lf_psi_values, psi_values


class DotDynamicsNormalRotationalCBF(DotDynamicsNormal):
    """
    Dynamics class that uses the Rotational CBF for obstacle avoidance.
    """

    def __init__(self, dt: float, p: float, lambda1: float, lambda2: float, u_max: float, p1: float, p2: float, 
                 initial_state: np.ndarray = np.array([80.0, 80.0, 0.0, 0.0, 0.0]),
                 constant_control: np.ndarray = None, k: float = 5):
        """
        Initialize the DotDynamics with Rotational CBF.
        
        Args:
            dt: Time step
            p: Radius parameter for the safety region
            lambda1: First eigenvalue (longitudinal direction)
            lambda2: Second eigenvalue (lateral direction), should be larger than lambda1
            u_max: Maximum control input magnitude
            p1: CBF parameter for barrier functions
            p2: CBF parameter for control derivation
            initial_state: Initial state vector
            constant_control: Constant control input
            k: Parameter for soft min calculation
        """
        super().__init__(dt=dt, initial_state=initial_state, constant_control=constant_control)
        self.rot_cbf = SoftMinLie([RotationalCBF2Order(p, lambda1, lambda2, u_max, p1)], k=k)
        self.p = p
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.u_max = u_max
        self.p2 = p2

    def perform_step(self, u: np.ndarray, observation: np.ndarray) -> np.ndarray:
        """
        Perform a step using the rotational CBF to ensure safety constraints.
        
        Args:
            u: Control input
            observation: Observation from the environment
            
        Returns:
            Next state
        """
        # Get constraints
        Lg_psi, Lf_psi, psi = self.rot_cbf.get_Lg_Lf_and_psi(observation)

        # Calculate optimal control
        u_safe = single_exponential_cbf_solver(u_ref=u, Lg_psi=Lg_psi, Lf_psi=Lf_psi, psi=psi, p1=self.p2)

        return super().perform_step(u_safe, observation)


if __name__ == "__main__":
    # Parameters
    U_MAX = 50
    DT = 1e-2
    P = 10  # Safety region parameter (similar to radius)
    LAMBDA1 = 1  # Longitudinal eigenvalue (in direction of motion)
    LAMBDA2 = 5  # Lateral eigenvalue (perpendicular to motion)
    P1 = 1  # CBF parameter
    P2 = 2  # CBF parameter for control
    K = 2  # Soft min parameter
    
    # Initialize dynamics with rotational CBF
    dynamics = DotDynamicsNormalRotationalCBF(
        dt=DT, 
        p=P, 
        lambda1=LAMBDA1, 
        lambda2=LAMBDA2, 
        u_max=U_MAX, 
        p1=P1, 
        p2=P2, 
        k=K
    )

    # Run the simulation
    runner(
         dynamics=dynamics,
         lidar_distance=130,
         lidar_num=64,
         u_max=U_MAX,
         render=True,
         initial_obstacles=10,
         #world_file=Path('worlds/tight_track.json'),
         num_steps=10000,
    )

import numpy as np
from pathlib import Path

from cbf_tools import FirstOrderGeneralLie, SoftMinLie, single_exponential_cbf_solver
from navigationgym import DotDynamicsNormal, runner

class RotationalCBF2Order(FirstOrderGeneralLie):
    """
    Implementation of the rotational CBF with an added velocity term as described in:

    \[
        h = r^T R \Lambda R^T r - \beta \frac{r^T}{\|r\|} \dot{r} - p^2
    \]

    Where:
    - \(R\) is a rotation matrix that rotates \(\|\dot{r}\|e_1\) into \(\dot{r}\)
    - \(\Lambda\) is a diagonal eigenvalue matrix with eigenvalues \(\lambda_1\) and \(\lambda_2\)
    - \(\beta\) is the coefficient of the additional velocity breaking term.
    """
    def __init__(self, p: float, lambda1: float, lambda2: float, u_max: float, beta: float):
        """
        Initialize the rotational CBF with the velocity term.
        
        Args:
            p: Radius parameter for the safety region.
            lambda1: First eigenvalue (longitudinal direction).
            lambda2: Second eigenvalue (lateral direction).
            u_max: Maximum control input.
            beta: Coefficient for the velocity breaking term.
        """
        self.p = p
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.u_max = u_max
        self.beta = beta
        # Lambda matrix as a diagonal matrix of eigenvalues
        self.Lambda = np.array([[lambda1, 0], [0, lambda2]])
        # Regularization constant to avoid numerical issues
        self.epsilon = 1e-6
    
    def get_rotation_matrix(self, vel_vector: np.ndarray) -> np.ndarray:
        """
        Computes the rotation matrix \(R\) that rotates \(\|\dot{r}\|e_1\) into \(\dot{r}\).
        
        Args:
            vel_vector: The velocity vector \([\dot{r}_1, \dot{r}_2]\)
            
        Returns:
            2x2 rotation matrix.
        """
        # Regularize the velocity to avoid numerical issues when it is close to zero.
        vel_reg = vel_vector + self.epsilon * np.array([1, 0])
        vel_norm = np.linalg.norm(vel_reg)
        
        R = np.array([
            [vel_reg[0] / vel_norm, -vel_reg[1] / vel_norm],
            [vel_reg[1] / vel_norm,  vel_reg[0] / vel_norm]
        ])
        
        return R
    
    def get_Lg_Lf_and_psi(self, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes \(L_g\), \(L_f\), and \(\psi\) based on the new CBF formulation with the velocity term.
        
        The new formulation is:
        
        \[
            h = r^T R \Lambda R^T r - \beta \frac{r^T}{\|r\|}\dot{r} - p^2,
        \]
        
        \[
            f = 2\dot{r}^T R \Lambda R^T r - \beta \frac{\|\dot{r}\|^2}{\|r\|} + \beta \frac{(r^T \dot{r})^2}{\|r\|^3},
        \]
        
        \[
            g = \frac{2}{\|\dot{r}\|^2} 
                \begin{bmatrix}
                    -\dot{r}_2 (r^T J R \Lambda R^T r) \\
                    \dot{r}_1 (r^T J R \Lambda R^T r)
                \end{bmatrix} - \beta\,\frac{r}{\|r\|}.
        \]
        
        Args:
            observation: The state observation vector.
            
        Returns:
            Tuple of \((L_g\psi, L_f\psi, \psi)\).
        """
        # Extract position and velocity from the observation.
        pos_vector = np.array([0, 0])  # Using the origin as the reference point.
        vel_vector = observation[3:5]
        circle_centers = observation[5:].reshape(-1, 2)
        
        psi_values = []
        Lg_psi_values = []
        Lf_psi_values = []
        
        for center in circle_centers:
            # Relative position vector: \(r =\) pos_vector - center
            r = pos_vector - center
            r_norm = np.linalg.norm(r) + self.epsilon
            r_unit = r / r_norm
            
            # Compute the rotation matrix \(R\).
            R = self.get_rotation_matrix(vel_vector)
            
            # Barrier function with an extra velocity term:
            # \(\displaystyle h = r^T R \Lambda R^T r - \beta (r/||r||)^T\dot{r} - p^2\)
            h = r @ R @ self.Lambda @ R.T @ r + self.beta * (r_unit @ vel_vector) - self.p**2
            
            # Compute the 2D skew-symmetric matrix \(J\).
            J = np.array([[0, -1], [1, 0]])
            
            # Compute the term \(r^T J R \Lambda R^T r\)
            r_JR_Lambda_RT_r = r @ J @ R @ self.Lambda @ R.T @ r
            
            # Compute the component corresponding to the original g term:
            vel_norm_squared = np.linalg.norm(vel_vector)**2 + self.epsilon
            g_R_term = 2 * np.array([
                -vel_vector[1] * r_JR_Lambda_RT_r,
                vel_vector[0] * r_JR_Lambda_RT_r
            ]) / vel_norm_squared
            
            # Incorporate the additional velocity term in g: \(-\beta \frac{r}{\|r\|}\)
            g = g_R_term + self.beta * (r / r_norm)
            
            # Compute \(L_f\psi\) (i.e. the drift term) using the new formulation:
            # \(\displaystyle f = 2\dot{r}^T R \Lambda R^T r - \beta \frac{\|\dot{r}\|^2}{\|r\|} + \beta \frac{(r^T \dot{r})^2}{\|r\|^3}\)
            lf_term1 = 2 * vel_vector @ R @ self.Lambda @ R.T @ r
            lf_term2 = self.beta * (np.linalg.norm(vel_vector)**2 / r_norm)
            lf_term3 = self.beta * ((r @ vel_vector)**2 / (r_norm**3))
            Lf = lf_term1 + lf_term2 - lf_term3
            
            psi_values.append(h)
            Lg_psi_values.append(g)
            Lf_psi_values.append(Lf)
        
        return np.array(Lg_psi_values), np.array(Lf_psi_values), np.array(psi_values)


class DotDynamicsNormalRotationalCBF(DotDynamicsNormal):
    """
    Dynamics class that uses the Rotational CBF with the velocity term for obstacle avoidance.
    """

    def __init__(self, dt: float, p: float, lambda1: float, lambda2: float, u_max: float, beta: float, p2: float, 
                 initial_state: np.ndarray = np.array([80.0, 80.0, 0.0, 0.0, 0.0]),
                 constant_control: np.ndarray = None, k: float = 5):
        """
        Initialize the DotDynamics with Rotational CBF including the velocity term.
        
        Args:
            dt: Time step.
            p: Radius parameter for the safety region.
            lambda1: First eigenvalue (longitudinal direction).
            lambda2: Second eigenvalue (lateral direction).
            u_max: Maximum control input magnitude.
            beta: Coefficient for the velocity breaking term.
            p2: CBF parameter used in control derivation.
            initial_state: Initial state vector.
            constant_control: Constant control input.
            k: Parameter for the soft-min calculation.
        """
        super().__init__(dt=dt, initial_state=initial_state, constant_control=constant_control)
        self.rot_cbf = SoftMinLie([RotationalCBF2Order(p, lambda1, lambda2, u_max, beta)], k=k)
        self.p = p
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.u_max = u_max
        self.beta = beta
        self.p2 = p2

    def perform_step(self, u: np.ndarray, observation: np.ndarray) -> np.ndarray:
        """
        Perform a step using the rotational CBF with velocity term to enforce safety constraints.
        
        Args:
            u: Control input.
            observation: Observation from the environment.
            
        Returns:
            Next state.
        """
        # Get the CBF constraints.
        Lg_psi, Lf_psi, psi = self.rot_cbf.get_Lg_Lf_and_psi(observation)

        # Compute the safe control input.
        u_safe = single_exponential_cbf_solver(u_ref=u, Lg_psi=Lg_psi, Lf_psi=Lf_psi, psi=psi, p1=self.p2)

        return super().perform_step(u_safe, observation)


if __name__ == "__main__":
    # Parameters
    U_MAX = 50
    DT = 1e-2
    P = 10             # Safety region parameter (similar to a radius)
    LAMBDA1 = 1        # Longitudinal eigenvalue (direction of motion)
    LAMBDA2 = 2        # Lateral eigenvalue (perpendicular to motion)
    BETA = 10           # Coefficient for the velocity breaking term
    P2 = 2             # CBF parameter used in control
    K = 0.5              # Soft-min parameter
    
    # Initialize dynamics with the updated rotational CBF
    dynamics = DotDynamicsNormalRotationalCBF(
        dt=DT, 
        p=P, 
        lambda1=LAMBDA1, 
        lambda2=LAMBDA2, 
        u_max=U_MAX, 
        beta=BETA,
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

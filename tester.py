import numpy as np
from triangles_nav_linear import DotDynamicsNormalTrianglesCBF
from soft_min_nav_linear import DotDynamicsNormalSoftMin
from navigationgym import runner, DotDynamicsNormal


if __name__ == "__main__":

    #tra

    U_MAX = 50.0

    dynamics = DotDynamicsNormal(dt=1e-2, control_size=U_MAX)
    
    #dynamics = DotDynamicsNormalSoftMin(dt=1e-2, radius=10, u_max=U_MAX, p1=2.55, p2=2,
    #                                    initial_state=np.array([80.0, 250.0, 0.0, 0.0, 0.0])) # 42 / 43
            

    # Run the simulation
    runner(
         dynamics=dynamics,
         lidar_distance=130,
         lidar_num=32,
         u_max=U_MAX,
         render=True,
         num_steps=3000,
         initial_obstacles=30,
    )
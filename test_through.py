
import numpy as np
from pathlib import Path

from triangles_nav_linear import DotDynamicsNormalTrianglesCBF
from soft_min_nav_linear import DotDynamicsNormalSoftMin
from navigationgym import runner

if __name__ == "__main__":

    results_base = Path("results")
    #results_base.mkdir(exist_ok=True)

    U_MAX = 50.0
    LIDAR_DISTANCE = 130
    LIDAR_NUM = 64
    DT = 1e-2

    for track, results_name, num_steps in [('worlds/tight_track.json', 'tight_track', 1000),
                                ('worlds/random_track_quadrant.json', 'random_track_quadrant', 2000),
                                ('worlds/zigzag_tight_track.json', 'zigzag_tight_track', 2300)]:
     
        dynamics_tri = DotDynamicsNormalTrianglesCBF(dt=DT, p1=2, p2=1.09, control_size=U_MAX,
                                                initial_state=np.array([65.0, 65.0, 0.0, 0.0, 0.0]),
                                                constant_control=np.array([U_MAX, U_MAX]))
        
        dynamics_soft = DotDynamicsNormalSoftMin(dt=DT, radius=0.1, u_max=U_MAX, p1=3, p2=2,
                                                 initial_state=np.array([65.0, 65.0, 0.0, 0.0, 0.0]),
                                                 constant_control=np.array([U_MAX, U_MAX]), k=0.1)

        runner(
            dynamics=dynamics_tri,
            lidar_distance=LIDAR_DISTANCE,
            lidar_num=LIDAR_NUM,
            u_max=U_MAX,
            render=True,
            num_steps=num_steps,
            results_path=results_base / Path(results_name+"_triangles"),
            world_file=Path(track),
            dt=DT
        )

        runner(
            dynamics=dynamics_soft,
            lidar_distance=LIDAR_DISTANCE,
            lidar_num=LIDAR_NUM,
            u_max=U_MAX,
            render=True,
            num_steps=num_steps,
            results_path=results_base / Path(results_name+"_soft"),
            world_file=Path(track),
            dt=DT
        )

    """track, results_name, num_steps = ('worlds/random_track_sparse.json', 'random_track_sparse', 1300)

    dynamics_tri = DotDynamicsNormalTrianglesCBF(dt=DT, p1=2, p2=1.09, control_size=U_MAX,
                                                initial_state=np.array([65.0, 65.0, 0.0, 0.0, 0.0]),
                                                constant_control=np.array([U_MAX, U_MAX]))
        
    dynamics_soft = DotDynamicsNormalSoftMin(dt=DT, radius=0.01, u_max=U_MAX, p1=3, p2=2,
                                                initial_state=np.array([65.0, 65.0, 0.0, 0.0, 0.0]),
                                                constant_control=np.array([U_MAX, U_MAX]), k=1)
    
    runner(
            dynamics=dynamics_tri,
            lidar_distance=LIDAR_DISTANCE,
            lidar_num=LIDAR_NUM,
            u_max=U_MAX,
            render=True,
            num_steps=num_steps,
            results_path=results_base / Path(results_name+"_triangles"),
            world_file=Path(track)
        )

    runner(
        dynamics=dynamics_soft,
        lidar_distance=LIDAR_DISTANCE,
        lidar_num=LIDAR_NUM,
        u_max=U_MAX,
        render=True,
        num_steps=num_steps,
        results_path=results_base / Path(results_name+"_soft"),
        world_file=Path(track)
    )"""
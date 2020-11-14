#import numpy as np
from helpfunctions import *
import time

if __name__ == "__main__":

    start = time.time()

    print("\n1. Load two point clouds and disturb initial pose.")
    source = read_point_cloud("PointClouds/gerador_1.ply")
    target = read_point_cloud("PointClouds/gerador_2.ply")
    trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0],
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]])
    source.transform(trans_init)                   
    source.transform(np.identity(4))
    #draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    draw_registration_result(source, target, np.identity(4))
    draw_registration_result_original_color(source, target, np.identity(4),"PointClouds/GERADOR_2_3.ply")

    voxel_size = 0.05 # means 5cm for the dataset
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size)

    #Fast Global Registration
    result_fast = execute_fast_global_registration(source_down, target_down,
            source_fpfh, target_fpfh, voxel_size)
    draw_registration_result(source_down, target_down,result_fast.transformation)

    # Colored Point Cloud Registration
    print("\n3. Applying colored point cloud registration")
    result_icp_color = registration_colored_icp(source_down, target_down,
                voxel_size, result_fast.transformation,
                ICPConvergenceCriteria(relative_fitness = 1e-6,
                relative_rmse = 1e-6, max_iteration = 50))     

    print(result_icp_color)
    
    draw_registration_result(source, target, result_icp_color.transformation)
    draw_registration_result_original_color(source, target, result_icp_color.transformation,"PointClouds/GERADOR_1_2.ply")

    print("\nTime %.3f sec.\n" % (time.time() - start))

  
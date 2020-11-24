from helpfunctions import *
import time
import numpy as np
import open3d
import copy

if __name__ == "__main__":

    start = time.time()

    print("\n1. Load two point clouds and disturb initial pose.\n")
    source = read_point_cloud("PointClouds/nuvem1.ply")
    target = read_point_cloud("PointClouds/nuvem2.ply")

    Method = 4  #  1 = Fast / 2 = Global / 3 = point-to-point ICP / 4 = point-to-plane

    trans_init = np.asarray([[-0.14, -0.02, 0.99, -3.66],
                            [-0.02, 1.00, -0.02,- 0.03],
                            [-0.9, -0.01, -0.14, 1.02],
                            [0.0, 0.0, 0.0, 1.0]])

    #trans_init = np.identity(4)

    threshold = 0.02
    print("\nInitial alignment")
    evaluation = evaluate_registration(source, target, threshold, trans_init)
    print(evaluation)

    #source.transform(trans_init)  # Uncomment if is method 1 or 2 
    draw_registration_result(source, target, trans_init)

    # Manual Adjustment
    #source.translate((3.1,0,6))
    #R = np.array([[0],[0.3],[0]],dtype = float)
    #source.rotate(R, True)

    voxel_size = 0.05 # means 5cm for the dataset

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    if Method == 1:
        
        #Fast Global Registration
        result_fast = execute_fast_global_registration(source_down, target_down,
                source_fpfh, target_fpfh, voxel_size)
        draw_registration_result(source_down, target_down,result_fast.transformation)
    
        print("\n",result_fast.transformation)

        result = result_fast.transformation

    if Method == 2:
        
        result_ransac = execute_global_registration(source_down, target_down,
                                                    source_fpfh, target_fpfh,
                                                    voxel_size)
        #print(result_ransac)
        draw_registration_result(source_down, target_down,
                                 result_ransac.transformation)

        result_icp = refine_registration(source, target, source_fpfh, target_fpfh,
                                         voxel_size,result_ransac)

        draw_registration_result(source_down, target_down,result_icp.transformation)

        print("\n",result_icp.transformation)

        result = result_icp.transformation

    if Method == 3:

        print("\n2. Apply point-to-point ICP")
        reg_p2p = registration_icp(
            source, target, threshold, trans_init,
            TransformationEstimationPointToPoint())
        print(reg_p2p)
        print("\nTransformation is:")
        print(reg_p2p.transformation)
        print("")
        draw_registration_result(source, target, reg_p2p.transformation)

        result = reg_p2p.transformation

    if Method == 4:

        print("\n2. Apply point-to-plane ICP")
        reg_p2l = registration_icp(
            source, target, threshold, trans_init,
            TransformationEstimationPointToPlane())
        print(reg_p2l)
        print("\nTransformation is:")
        print(reg_p2l.transformation)
        print("")
        draw_registration_result(source, target, reg_p2l.transformation)

        result = reg_p2l.transformation


    #--------------------------------------------------------------------------------------------------------------------------
    
    # Colored Point Cloud Registration
    print("\n3. Applying colored point cloud registration")
    result_icp_color = registration_colored_icp(source_down, target_down,
                voxel_size, result,
                ICPConvergenceCriteria(relative_fitness = 1e-6,
                relative_rmse = 1e-6, max_iteration = 50))
    
    print("\n", result_icp_color.transformation, "\n")

    #result_multi = result_icp_color.transformation*result_fast.transformation;

    #print("4. Multiplication\n")

    #print(result_multi, "\n")

    #print(result_icp_color)
    
    draw_registration_result(source, target, result_icp_color.transformation)
    draw_registration_result_original_color(source, target, result_icp_color.transformation,"Results/Juncao.ply")

    np.savetxt('Results/transformada.txt',result_icp_color.transformation)

    if ((time.time() - start) <= 60):

       print("\nTime %.3f sec.\n" % (time.time() - start))
    
    else:

       print("\nTime %.3f min.\n" % ((time.time() - start)/60))

  
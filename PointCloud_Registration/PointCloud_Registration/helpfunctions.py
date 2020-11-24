# Help Functions

from open3d import *
import copy


def preprocess_point_cloud(ply, voxel_size):
    #print("\n:: Downsample with a voxel size %.3f." % voxel_size)
    ply_down = voxel_down_sample(ply, voxel_size)

    radius_normal = voxel_size * 2
    #print(":: Estimate normal with search radius %.3f." % radius_normal)
    estimate_normals(ply_down, KDTreeSearchParamHybrid(
            radius = radius_normal, max_nn = 30))

    radius_feature = voxel_size * 5 #era 5
    #print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    ply_fpfh = compute_fpfh_feature(ply_down,
            KDTreeSearchParamHybrid(radius = radius_feature, max_nn = 50)) #era 100
    return ply_down, ply_fpfh

#-----------------------------------------------------------------------------------------------------------------

def execute_fast_global_registration(source_down, target_down,
        source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print("\n2. Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result = registration_fast_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            FastGlobalRegistrationOption(
            maximum_correspondence_distance = distance_threshold))

    return result

#--------------------------------------------------------------------------------------------------------------

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):

    distance_threshold = voxel_size * 1.5

    print("\n2. Apply global registration with distance threshold %.3f" \
        % distance_threshold)
    
    print("\n:: RANSAC registration on downsampled point clouds.")
    #print("   Since the downsampling voxel size is %.3f," % voxel_size)
    #print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        TransformationEstimationPointToPoint(False), 4, [
            CorrespondenceCheckerBasedOnEdgeLength(0.9),
            CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], RANSACConvergenceCriteria(4000000, 500))
    return result

#------------------------------------------------------------------------------------------------------------------

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac):

   distance_threshold = voxel_size * 0.4
   print("\n:: Point-to-plane ICP registration is applied on original point")
   print("   clouds to refine the alignment. This time we use a strict")
   print("   distance threshold %.3f." % distance_threshold)
   result = registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        TransformationEstimationPointToPlane())
    
   
   return result

#-------------------------------------------------------------------------------------------------------------------
#                                                    DRAW
#-------------------------------------------------------------------------------------------------------------------

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    draw_geometries([source_temp, target_temp])


def draw_registration_result_original_color(source, target, transformation,name):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation) 
    newpointcloud = source_temp + target         
    draw_geometries([newpointcloud])             
    write_point_cloud (name, newpointcloud)  


    
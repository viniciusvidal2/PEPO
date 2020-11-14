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

    radius_feature = voxel_size * 5
    #print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    ply_fpfh = compute_fpfh_feature(ply_down,
            KDTreeSearchParamHybrid(radius = radius_feature, max_nn = 100))
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


    
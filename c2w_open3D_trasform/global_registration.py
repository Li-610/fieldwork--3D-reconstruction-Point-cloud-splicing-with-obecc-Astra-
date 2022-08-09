import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import open3d as o3d
import numpy as np


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(source, target, voxel_size):
    print(":: Load two point clouds and disturb initial pose.")

    trans_init = np.eye(4)
    source.transform(trans_init)

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source_down, target_down, source_fpfh, target_fpfh

#little slower but result better
def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    transformation = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    
    return transformation

#faster but get a worse result
'''
def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    print(":: Apply fast global registration with distance threshold %.3f" \
            % distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))

    print(result)

    return result
'''

if __name__ == '__main__':
    from mix_PointCloud import read_ply
    from creat_c2w import *

    voxel_size = 100
    points = read_ply('data/dinning_room2')

    c2w = creat_c2w('data/dinning_room2', points[0], 21)
    target = points[0].transform(c2w)

    c2w = creat_c2w('data/dinning_room2', points[1], 22)
    source = points[1].transform(c2w)

    print('===============')
    print(':: Start data prepare')
    source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source, target, voxel_size)

    print(':: Global registration start')
    result = execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size)
    print(':: Global registration finish\n\n')
    
    source.transform(result.transformation)
    o3d.io.write_point_cloud('log/global_registration/target.ply', target)
    o3d.io.write_point_cloud('log/global_registration/source.ply', source)

    o3d.visualization.draw_geometries([target, source], width = 1000, height = 600)
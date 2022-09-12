import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import open3d as o3d
from global_registration import *


def ICP(source, target, trans_init):
    threshold = 50

    print(':: Initial alignment: ')
    evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, trans_init)
    print(evaluation)

    print(':: Initial tansformation matrix:')
    print(trans_init)

    print(':: Apply point to point ICP to point cloud ')
    icp_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 5000)
    )

    print('=======================')
    print(icp_p2p)
    print(':: Transformation matrix is:')
    print(icp_p2p.transformation)

    return icp_p2p.transformation

def colored_icp(target, source, trans_init):
    voxel_radius = [20, 10, 5]
    max_iter = [50, 20, 14]
    current_transformation = trans_init
    print("3. Colored point cloud registration")
    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        print([iter, radius, scale])

        print("3-1. Downsample with a voxel size %.2f" % radius)
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)

        print("3-2. Estimate normal.")
        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn = 30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn = 30))

        print("3-3. Applying colored point cloud registration")
        result_icp = o3d.pipelines.registration.registration_colored_icp(
            source_down, target_down, radius, current_transformation,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                              relative_rmse=1e-6,
                                                              max_iteration=iter))
        current_transformation = result_icp.transformation
        print(result_icp)

    return current_transformation

if __name__ == '__main__':
    from creat_c2w import *
    #from mix_PointCloud import *

    voxel_size = 100
    points, num = read_ply('data/meeting_room60')
    
    a = 20
    c = a + 10
    #c = a
    target = points[a-1].transform(creat_c2w('data/meeting_room60', points[a-1], a))
    
    #target = o3d.io.read_point_cloud('log/ICP/flower_bed/result_%a.ply'%c, format = 'ply')

    source = points[c-1].transform(creat_c2w('data/meeting_room60', points[c-1], c))

    #target = o3d.io.read_point_cloud('log/ICP/flower_bed/point_cloud/result_1.ply', format = 'ply')

    #source = o3d.io.read_point_cloud('log/ICP/flower_bed/point_cloud/result_2.ply', format = 'ply')

    source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source, target, voxel_size)
    result = execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size)
    print('Global registration finish\n\n')

    source = source.voxel_down_sample(10)
    target = target.voxel_down_sample(10)

    print('Iterative Closest Point registration start')
    trans = ICP(source, target, result.transformation)

    #print('-----------------------------')
    #print(np.asarray(pcd.points))

    source.transform(trans)

    o3d.io.write_point_cloud('log/meeting_room/ICP/target.ply', target)
    o3d.io.write_point_cloud('log/meeting_room/ICP/source.ply', source)

    o3d.io.write_point_cloud('log/meeting_room/ICP/point_cloud_%a.ply'%a, source)
    o3d.io.write_point_cloud('log/meeting_room/ICP/result_%a.ply'%a, target + source)

    o3d.visualization.draw_geometries([target, source], width = 1000, height = 600)
    write_ply([target, source])
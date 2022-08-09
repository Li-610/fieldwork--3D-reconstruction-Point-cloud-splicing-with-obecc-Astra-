import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import open3d as o3d
from global_registration import *


def ICP(source, target, trans_init):
    threshold = 40

    print(':: Initial alignment: ')
    evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, trans_init)
    print(evaluation)

    print(':: Initial tansformation matrix:')
    print(trans_init)

    print(':: Apply point to point ICP to point cloud ')
    icp_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 1000)
    )
            
    print('=======================')
    print(icp_p2p)
    print(':: Transformation matrix is:')
    print(icp_p2p.transformation)

    return icp_p2p

if __name__ == '__main__':
    from creat_c2w import *
    from mix_PointCloud import write_ply

    voxel_size = 100
    points = read_ply('data/dinning_room2')

    target = points[2]

    source = points[3]

    source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source, target, voxel_size)
    result = execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size)
    print('Global registration finish\n\n')

    target = target.voxel_down_sample(10)
    source = source.voxel_down_sample(10)

    print('Iterative Closest Point registration start')
    icp_p2p = ICP(source, target, result.transformation)

    #print('-----------------------------')
    #print(np.asarray(pcd.points))

    source.transform(icp_p2p.transformation)

    o3d.io.write_point_cloud('log/ICP/target.ply', target)
    o3d.io.write_point_cloud('log/ICP/source.ply', source)

    o3d.visualization.draw_geometries([target, source], width = 1000, height = 600)
    write_ply([target, source])
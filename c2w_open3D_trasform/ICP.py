import open3d as o3d
from creat_c2w import *
from mix_PointCloud import *

def ICP(source, target, trans_init):
    threshold = 0.1

    print(':: Initial alignment: ')
    evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, trans_init)
    print('   ' + evaluation)
    print(':: Apply point to point ICP to point cloud ')

    print(':: Initial tansformation matrix:')
    print('   ' + trans_init)

    icp_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 600)
    )
            
    print('=======================')
    print(icp_p2p)
    print(':: Transformation matrix is:')
    print('   ' + icp_p2p.transformation)

    source.transform(icp_p2p.transformation)
    print(':: Finish transformation for point cloud')
    print('=======================')

    return source + target

if __name__ == '__main__':
    points = read_ply('data/trash_bin')

    c2w = creat_c2w('data/trash_bin', points[0], 21)
    points[0].transform(c2w)
    target = points[0].voxel_down_sample(voxel_size = 0.05)

    c2w = creat_c2w('data/trash_bin', points[1], 22)
    #points[1].transform(c2w)
    #pcd = points[0] + points[1]
    #pcd = ICP(points[1], points[0], np.eye(4))

    source = points[1].voxel_down_sample(voxel_size = 0.05)
    pcd = ICP(source, target, c2w)

    #print('-----------------------------')
    #print(np.asarray(pcd.points))


    o3d.visualization.draw_geometries([pcd], width = 1000, height = 600)
    write_ply(pcd, 'trash_bin')
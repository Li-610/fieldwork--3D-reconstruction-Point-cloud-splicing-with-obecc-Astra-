from plyfile import *
import cv2
import open3d as o3d
import pdb
import os
import numpy as np
from creat_c2w import *
from global_registration import *
from ICP import *

def read_ply(path, fileName = 'PointCloud'):
    file_path = os.path.join(path, fileName)

    poit_cloud = []
    pcd_file = os.listdir(file_path)

    for file in pcd_file:
        file_name = os.path.join(file_path,file)
        poit_cloud.append(o3d.io.read_point_cloud(file_name, format = 'ply'))

    return poit_cloud

def registration(target, source, voxel_size = 100):
    print(':: Start data prepare')
    source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source, target, voxel_size)

    print(':: Global registration start\n\n')
    result = execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size)
    print(':: Global registration finish\n\n')

    target_down = target.voxel_down_sample(10)
    source_down = source.voxel_down_sample(10)
    print(':: Iterative Closest Point registration start')
    icp_p2p = ICP(source_down, target_down, result.transformation)
    print(':: Iterative Closest Point registration finish')

    return icp_p2p.transformation

def tran_PointCloud(path, target = None, source = None):
    points = read_ply(path)
    
    cloud_num = 21
    for point in points:

        #get camera to world matrix
        c2w = creat_c2w(path, point,cloud_num)

        #transform point cloud from camera coordinate system to world coordinate system
        point.transform(c2w)
        source = point
        
        if target == None:
            target = points[0]
        else:
            transformation = registration(target, source)
            point.transform(transformation)
            target += point

        cloud_num += 1

    return points

def write_ply(points, file_name = 'result.ply', save_path='log'):
    point_cloud = None

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    file_path = os.path.join(save_path,file_name)
    
    for point in points:
        if point_cloud is None: point_cloud = point
        point_cloud += point
    o3d.io.write_point_cloud(file_path, point_cloud)

if __name__ == '__main__':
    pcd = tran_PointCloud('data/dinning_room2')
    for i in range(1,21):
        o3d.io.write_point_cloud('log/dinning_room2/'+str(i)+'.ply', pcd[i-1])
    write_ply(pcd)
    
    o3d.visualization.draw_geometries(pcd, width = 1000, height = 600)
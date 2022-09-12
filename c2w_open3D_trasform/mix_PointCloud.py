import numbers
from plyfile import *
import open3d as o3d
import pdb
import os
import numpy as np
import copy
from creat_c2w import *
from global_registration import *
from ICP import *

def read_ply(path, fileName = 'PointCloud'):
    file_path = os.path.join(path, fileName)
    numbers = []

    poit_cloud = []
    pcd_file = os.listdir(file_path)
    pcd_file.sort(key=lambda x:int(x[:-4]))

    for file in pcd_file:
        numbers.append(file.split('.')[0])
        file_name = os.path.join(file_path,file)
        poit_cloud.append(o3d.io.read_point_cloud(file_name, format = 'ply'))

    return poit_cloud, numbers

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
    transformation = ICP(source_down, target_down, result.transformation)
    print(':: Iterative Closest Point registration finish')

    return transformation

def tran_PointCloud(path, target = None, source = None):
    points, numbers = read_ply(path)
    
    count = 0
    for point in points:

        cloud_num = int(numbers[count])
        #get camera to world matrix
        c2w = creat_c2w(path, point, cloud_num)

        point.transform([[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

        #transform point cloud from camera coordinate system to world coordinate system
        point.transform(c2w)
        source = copy.deepcopy(point)
        
        if target == None:
            target = copy.deepcopy(point)
        else:
            transformation = registration(target, source)
            point.transform(transformation)
            target = copy.deepcopy(point)

        count += 1
        
    return points

def write_ply(points, file_name = 'result.ply', save_path = 'log'):
    point_cloud = None

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    file_path = os.path.join(save_path,file_name)
    
    for point in points:
        point.voxel_down_sample(100)
        if point_cloud is None: point_cloud = copy.deepcopy(point)
        point_cloud += point
    point_cloud.voxel_down_sample(1000) 
    o3d.io.write_point_cloud(file_path, point_cloud)

if __name__ == '__main__':
    path = 'log/meeting_room'

    pcd = tran_PointCloud('data/meeting_room')

    write_ply(pcd, save_path = path)

    for i in range(len(pcd)):
        o3d.io.write_point_cloud('log/meeting_room/'+str(i)+'.ply', pcd[i])
    
    o3d.visualization.draw_geometries(pcd, width = 1000, height = 600)
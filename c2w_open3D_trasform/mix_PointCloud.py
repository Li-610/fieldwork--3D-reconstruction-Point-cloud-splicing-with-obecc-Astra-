from plyfile import *
import cv2
import open3d as o3d
import pdb
import os
import numpy as np
from creat_c2w import *

def read_ply(path, fileName = 'PointCloud'):
    file_path = os.path.join(path, fileName)

    poit_cloud = []
    pcd_file = os.listdir(file_path)

    for file in pcd_file:
        file_name = os.path.join(file_path,file)
        poit_cloud.append(o3d.io.read_point_cloud(file_name, format = 'ply'))

    return poit_cloud

def tran_PointCloud(path):
    points = read_ply(path)
    
    cloud_num = 21
    for point in points:
        #get camera to world matrix
        c2w = creat_c2w(path, point,cloud_num)

        #transform point cloud from camera coordinate system to world coordinate system
        point.transform(c2w)

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
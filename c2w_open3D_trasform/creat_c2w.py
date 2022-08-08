from calendar import c
import open3d as o3d
import pdb
import os
import csv
from itertools import count
import numpy as np

def image_data_loader(path, image_ID):
    read_file = []
    file_path = os.path.join(path,'images.txt')

    f = open(file_path,'r')
    d = csv.reader(f)

    count = 0
    for line in d:
        if count%2 == 0 and count > 3:
            read_file.append(line)
        count += 1
    f.close

    for data in read_file:
        if image_ID == int(data[0].split(' ')[0]):
            qw = float(data[0].split(' ')[1])
            qx = float(data[0].split(' ')[2])
            qy = float(data[0].split(' ')[3])
            qz = float(data[0].split(' ')[4])

            tx = float(data[0].split(' ')[5])
            ty = float(data[0].split(' ')[6])
            tz = float(data[0].split(' ')[7])

    return [qw, qx, qy, qz], [tx, ty, tz]

def creat_c2w(path, point, ID):
    qvec, t = image_data_loader(path, ID)

    extrinsics = point.get_rotation_matrix_from_quaternion(qvec)
    
    c2w = np.matrix(extrinsics)
    c2w = np.column_stack((c2w, t))
    c2w = np.row_stack((c2w,[0,0,0,1]))

    return c2w

if __name__ == '__main__':
    #read_files()

    path = 'data/dinning_room2'
    
    poit_cloud = []
    pcd_file = os.listdir(path)

    for file in pcd_file:
        file_name = os.path.join(path,file)
        poit_cloud.append(o3d.io.read_point_cloud(file_name))
    
    
    point = poit_cloud[0]

    c2w = creat_c2w(path, point,23)
    print(image_data_loader(path, 23))
    print('-------------')
    print(c2w)

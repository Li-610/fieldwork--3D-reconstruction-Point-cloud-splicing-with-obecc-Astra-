import open3d as o3d
import numpy as np
import os
 
pcd_load = o3d.io.read_point_cloud("data/dinning_room2/PointCloud/RGBDPoints_02.ply")
pcd_down = pcd_load.voxel_down_sample(100)
o3d.visualization.draw_geometries([pcd_down], width = 1000, height = 600)
# convert Open3D.o3d.geometry.PointCloud to numpy array

'''xyz_load = np.asarray(pcd_load.points)
print(xyz_load)

print(os.environ["CUDA_VISIBLE_DEVICES"])'''
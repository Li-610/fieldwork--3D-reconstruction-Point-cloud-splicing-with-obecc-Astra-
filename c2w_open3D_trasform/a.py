import open3d as o3d
import numpy as np
 
 
pcd_load = o3d.io.read_point_cloud("log/dinning_room2/2.ply")
# convert Open3D.o3d.geometry.PointCloud to numpy array
xyz_load = np.asarray(pcd_load.points)
print(xyz_load)
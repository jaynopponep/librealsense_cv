import cv2
import numpy as np
import open3d as o3d

depth_image = cv2.imread("Pointcloud/orange_1_1_1_depthcrop.png", cv2.IMREAD_UNCHANGED)  
mask = cv2.imread("Pointcloud/orange_1_1_1_maskcrop.png", cv2.IMREAD_GRAYSCALE)  

with open("Pointcloud/orange_1_1_1_loc.txt", "r") as f:

    loc_x, loc_y = map(int, f.read().strip().split(","))  

fx = 525.0  
fy = 525.0 
cx = 320.0
cy = 240.0

h, w = depth_image.shape
points = []
colors = []

rgb_image = cv2.imread("Pointcloud/orange_1_1_1_crop.png")  

for v in range(h):
    for u in range(w):
        if mask[v, u] > 0:  
            Z = depth_image[v, u] / 1000.0  
            if Z > 0:
                u_full = u + loc_x
                v_full = v + loc_y
                
                X = (u_full - cx) * Z / fx
                Y = (v_full - cy) * Z / fy

                points.append((X, Y, Z))
                colors.append(rgb_image[v, u] / 255.0)  

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.array(points))
pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

o3d.io.write_point_cloud("orange_pointcloud.ply", pcd)

o3d.visualization.draw_geometries([pcd])

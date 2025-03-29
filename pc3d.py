import pyrealsense2 as rs
import numpy as np
import cv2
import pandas as pd
import open3d as o3d
import time

pipeline = rs.pipeline() #realsense pipeline 
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) #config for depth stream
pipeline.start(config)
print("Streaming depth. Press 'q' to capture a frame or wait 5 seconds...") #the user is informed that the depth stream is active and can be captured by pressing 'q' or waiting 5 seconds

#openCV window for depth display
cv2.namedWindow('Depth Stream', cv2.WINDOW_AUTOSIZE) #opencv window for depth stream display
start_time = time.time()
depth_frame = None

#stream loop to display depth and wait for quit or timeout
while True:
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    if not depth_frame: #if no depth frame is captured, continue to the next iteration
        continue
    #lidar depth pipeline + rgb color map
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_color = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)  
    cv2.imshow('Depth Stream', depth_color) #display depth stream in opencv window
    depth_value = depth_image[depth_image.shape[0]//2, depth_image.shape[1]//2]
    depth_in_meters = depth_value * 0.001  # from mm to meters 
    cv2.putText(depth_color, f"Center Depth: {depth_in_meters:.2f}m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    #break if 'q' pressed or 5 seconds have passed
    if (cv2.waitKey(1) & 0xFF == ord('q')) or (time.time() - start_time > 5):
        break

pipeline.stop()
cv2.destroyAllWindows()

#if a depth frame was captured, processs it into a point cloud
if depth_frame:
    pc = rs.pointcloud()
    points = pc.calculate(depth_frame)  #calculate point cloud from depth frame
    verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)  #convert to numpy array
    verts = verts[verts[:, 2] > 0]
    #save the point cloud to a file
    z_min, z_max = 0.2, 1.2  # meters
    mask = (verts[:, 2] > z_min) & (verts[:, 2] < z_max)
    filtered = verts[mask]

    if filtered.shape[0] == 0:
        print("No valid points in z-range. Try moving your hand further from the camera.")
        exit()

    #saving the point cloud to parquet file
    df = pd.DataFrame(filtered, columns=['x', 'y', 'z'])
    df['landmark_index'] = df.index
    df['frame'] = 0 #assuming single frame capture
    df.to_parquet("point_cloud.parquet", index=False) #save to parquet

    #open3d stuff to view the saved point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered)
    o3d.io.write_point_cloud("output.pcd", pcd) #save to pcd
    o3d.visualization.draw_geometries([pcd])
else:
    print("No depth frame captured. Exiting.")
    exit(1)
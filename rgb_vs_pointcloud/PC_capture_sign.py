import pyrealsense2 as rs
import numpy as np
import cv2
import pandas as pd
import open3d as o3d
import time
import openpifpaf #python 3.7 alteranative for mediapipe
import os
from scipy.spatial import cKDTree


POSE_KPTS = list(range(0, 17)) #define keypoint indices for different body parts
FACE_KPTS = list(range(17, 85))
LEFT_HAND_KPTS = list(range(85, 106))
RIGHT_HAND_KPTS = list(range(106, 127))

#predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k16-wholebody') 
predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k30-wholebody')
#openpifpaf predictor for whole body pose estimation, using a pre-trained model. 
#this will be used to detect keypoints in the point cloud data. mediapipe only work on 3.9+, so this is a workaround for python 3.7.

pipeline = rs.pipeline() #realsense pipeline 
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) #config for depth stream
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # added color stream for visualization purposes, this will help in debugging and visualizing the depth data alongside the point cloud data.
pipeline.start(config)
print("Streaming depth. Press 'q' to capture a frame or wait 5 seconds...") #the user is informed that the depth stream is active and can be captured by pressing 'q' or waiting 5 seconds

#openCV window for depth display
cv2.namedWindow('Depth Stream', cv2.WINDOW_AUTOSIZE) #opencv window for depth stream display
start_time = time.time()
depth_frame = None
color_frame = None  #also initialize color frame for openpifpaf purposes
landmark_df = None  

#stream loop to display depth and wait for quit or timeout
while True:
    frames = pipeline.wait_for_frames()
    align = rs.align(rs.stream.color) #align the depth stream to the color stream, this will help in aligning the depth data with the color data for better visualization and keypoint detection later on
    frames = align.process(frames)
    
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()  
    if not depth_frame: #if no depth frame is captured, continue to the next iteration
        continue
    #lidar depth pipeline + rgb color map
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_color = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.3), cv2.COLORMAP_JET)  
    cv2.imshow('Depth Stream', depth_color) #display depth stream in opencv window
    depth_value = depth_image[depth_image.shape[0]//2, depth_image.shape[1]//2]
    depth_in_meters = depth_value * 0.001  # from mm to meters 
    cv2.putText(depth_color, f"Center Depth: {depth_in_meters:.2f}m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    #break if 'q' pressed or 5 seconds have passed
    if (cv2.waitKey(1) & 0xFF == ord('q')) or (time.time() - start_time > 5):
        break

        #now for openpifpaf plotting keypoints
        # run openpifpaf
    color_image = np.asanyarray(color_frame.get_data())
    image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    predictions, _, _ = predictor.numpy_image(image_rgb)
    if predictions:
        keypoints = predictions[0].data
        rows = []
        for i, (x, y, conf) in enumerate(keypoints):
            if conf < 0.05:
                continue
            x_pix, y_pix = int(x), int(y)
            x_norm, y_norm = x / color_image.shape[1], y / color_image.shape[0]

            if i in POSE_KPTS:
                typ = 'pose'
                idx = i
            elif i in FACE_KPTS:
                typ = 'face'
                idx = i - 17
            elif i in LEFT_HAND_KPTS:
                typ = 'left_hand'
                idx = i - 85
            elif i in RIGHT_HAND_KPTS:
                typ = 'right_hand'
                idx = i - 106
            else:
                typ = 'other'
                idx = i
            z = depth_frame.get_distance(x_pix, y_pix)
            if z == 0:
                continue
            rh_ids = [row['landmark_index'] for row in rows if row['type'] == 'right_hand']
            print("Right hand indices:", rh_ids) #for debugging purposes, print the right hand indices to see if they are being captured correctly
            rows.append({'type': typ, 'landmark_index': idx, 'x': x_norm, 'y': y_norm, 'z': z, 'frame': 0})
        landmark_df = pd.DataFrame(rows)
        landmark_df.to_parquet("test_landmarks.parquet", index=False) #save the landmark data to a parquet file, this will be used for further processing and classification
        #print("Saved 2D+depth landmark data to landmarks.parquet")
        
pipeline.stop() 
cv2.destroyAllWindows()

#if a depth frame was captured, processs it into a point cloud
if depth_frame:
    pc = rs.pointcloud()
    points = pc.calculate(depth_frame)  #calculate point cloud from depth frame
    verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)  #convert to numpy array
    verts = verts[verts[:, 2] > 0]
    #save the point cloud to a file
    x_min, x_max = -0.25, 0.25   # narrower horizontally (remove edge clutter)
    y_min, y_max = -0.2, 0.4   # only upper body and arms
    z_min, z_max = 0.4, 1.1    # exclude very close and very far objects


    mask = (
        (verts[:, 0] > x_min) & (verts[:, 0] < x_max) &
        (verts[:, 1] > y_min) & (verts[:, 1] < y_max) &
        (verts[:, 2] > z_min) & (verts[:, 2] < z_max)
    )

    filtered = verts[mask]

    if filtered.shape[0] == 0:
        print("No valid points in z-range. Try moving your hand further from the camera.")
        exit()

    #saving the point cloud to parquet file
    df = pd.DataFrame(filtered, columns=['x', 'y', 'z'])
    df['landmark_index'] = df.index
    df['frame'] = 0 #assuming single frame capture
    points_np = df[['x', 'y', 'z']].to_numpy() #convert to numpy array so classifier can use it
    #df.to_parquet("point_cloud.parquet", index=False) #save to parquet

    pc_tree = cKDTree(df[['x', 'y', 'z']].values)
    lm_coords = landmark_df[['x', 'y', 'z']].values
    _, indices = pc_tree.query(lm_coords, k=1)

    type_map = {'pose': 0, 'face': 1, 'left_hand': 2, 'right_hand': 3}
    type_ids = landmark_df['type'].map(type_map).values

    labeled_df = df.iloc[indices].copy()
    labeled_df['type'] = landmark_df['type'].values
    labeled_df['landmark_index'] = landmark_df['landmark_index'].values
    labeled_df['type_id'] = type_ids
    labeled_df = labeled_df.dropna(subset=['type_id'])

    labeled_df.to_parquet("labeled_point_cloud.parquet", index=False)
    print("Saved labeled point cloud to labeled_point_cloud.parquet")


    #open3d stuff to view the saved point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np) #convert to open3d point cloud
    o3d.io.write_point_cloud("output.pcd", pcd) #save to pcd
    o3d.visualization.draw_geometries([pcd])
else:
    print("No depth frame captured. Exiting.")
    exit(1)
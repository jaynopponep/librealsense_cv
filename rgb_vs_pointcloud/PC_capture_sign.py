import time
import cv2
import numpy as np
import pyrealsense2 as rs
import openpifpaf  # python 3.7 alternative for MediaPipe
import torch
import open3d as o3d
import pandas as pd

torch.backends.cudnn.benchmark = True

# define keypoint indices for different body parts
POSE_KPTS       = list(range(0, 17))
FOOT_KPTS       = list(range(17, 23))    # feet (optional)
FACE_KPTS       = list(range(23, 91))
LEFT_HAND_KPTS  = list(range(91, 112))
RIGHT_HAND_KPTS = list(range(112, 133))

# load whole‑body model
predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k30-wholebody')

# RealSense setup
pipeline = rs.pipeline()
config   = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile  = pipeline.start(config)
align    = rs.align(rs.stream.color)

# ─── HERE: grab intrinsics for deprojection ─────────────────────────────────
color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
intrinsics   = color_stream.get_intrinsics()
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale  = depth_sensor.get_depth_scale()

print("Streaming depth & color. Press 'q' to quit or wait 30 seconds…")

# position OpenCV windows
cv2.namedWindow('Depth Stream',   cv2.WINDOW_AUTOSIZE)
cv2.moveWindow('Depth Stream',   50,  50)
cv2.namedWindow('PifPaf Overlay', cv2.WINDOW_AUTOSIZE)
cv2.moveWindow('PifPaf Overlay', 750,  50)

start = time.time()
depth_frame = None
landmark_rows = []
frame_idx=0

# MAIN LOOP: show live depth + PifPaf overlay
while True:
    frames      = pipeline.wait_for_frames()
    aligned     = align.process(frames)
    depth_frame = aligned.get_depth_frame()
    color_frame = aligned.get_color_frame()
    if not depth_frame or not color_frame:
        continue

    frame_idx += 1

    # build depth heatmap
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_color = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_image, alpha=0.3),
        cv2.COLORMAP_JET
    )

    # get color frame & run PifPaf
    color_image = np.asanyarray(color_frame.get_data())
    frame_bgr   = cv2.flip(color_image, 1)
    frame_rgb   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, _     = frame_bgr.shape

    try:
        preds, _, _ = predictor.numpy_image(frame_rgb)
    except Exception as e:
        print("PifPaf error:", e)
        continue

    # draw face & hand keypoints
    if preds:
        kp = preds[0].data  # shape (133,3)
        for i, (x, y, conf) in enumerate(kp):
            if conf < 0.05:
                continue
            if i in POSE_KPTS:
                color = (0, 0, 255)    # red for body
                typ   = 'pose'
                idx   = i
            elif i in FACE_KPTS:
                color = (0, 140, 255)
                typ = 'face'       
                idx = i - 23
            elif i in LEFT_HAND_KPTS:
                color = (255, 0, 0)
                typ = 'left_hand'  
                idx = i - 91
            elif i in RIGHT_HAND_KPTS:
                color = (0, 255, 0)
                typ = 'right_hand'
                idx = i - 112
            else:
                continue
            cv2.circle(frame_bgr, (int(x), int(y)), 4, color, -1)

            # ─── HERE: deproject pixel (x,y) + depth → real‑world (X,Y,Z) in meters ──────────────────
            depth_m = depth_frame.get_distance(int(x), int(y))
            if depth_m == 0:
                continue
            point3d = rs.rs2_deproject_pixel_to_point(
                intrinsics,
                [int(x), int(y)],
                depth_m
            )
            X_m, Y_m, Z_m = point3d  # in meters

            # store deprojected coordinates
            landmark_rows.append({
                'frame':          frame_idx,
                'type':           typ,
                'landmark_index': idx,
                'x':              X_m,
                'y':              Y_m,
                'z':              Z_m,
            })

    # show both windows
    cv2.imshow('Depth Stream',   depth_color)
    cv2.imshow('PifPaf Overlay', frame_bgr)

    key = cv2.waitKey(1)
    if key == ord('q') or (time.time() - start) > 20:
        break

# cleanup OpenCV & RealSense
pipeline.stop()
cv2.destroyAllWindows()

if landmark_rows:
    df = pd.DataFrame(landmark_rows)
    # change filename as you like
    df.to_parquet("3d_landmarks.parquet", index=False)
    print(f"✅ Saved {len(df)} landmarks → 3d_landmarks.parquet")
# AFTER CAPTURE: show final point cloud in Open3D
if depth_frame and landmark_rows:
    pc     = rs.pointcloud()
    points = pc.calculate(depth_frame)
    verts  = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
    verts  = verts[verts[:, 2] > 0]

    # half‑scale cropping around you (±0.5 m horizontally, 0.2–1.5 m depth)
    x_min, x_max = -0.5,  0.5
    y_min, y_max =  -1.20,  0.20 
    z_min, z_max =  0.2,  1.5
    mask = (
        (verts[:, 0] > x_min) & (verts[:, 0] < x_max) &
        (verts[:, 1] > y_min) & (verts[:, 1] < y_max) &
        (verts[:, 2] > z_min) & (verts[:, 2] < z_max)
    )
    filtered = verts[mask]

    if filtered.shape[0] == 0:
        print("No valid points in range. Exiting.")
        exit(1)


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered)
    o3d.visualization.draw_geometries(
        [pcd],
        window_name='Captured PointCloud (Half‑scale)'
    )
else:
    print("No depth frame captured. Exiting.")
    exit(1)
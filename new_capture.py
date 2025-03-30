import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pyrealsense2 as rs
import time
import os

mp_holistic = mp.solutions.holistic


def get_3d_landmarks(results, color_shape, depth_frame, depth_intrin):
    h, w = color_shape[:2]
    landmarks_3d = []

    def extract_landmarks(landmark_list, expected_count):
        coords = []
        if landmark_list:
            for lm in landmark_list.landmark:
                px, py = int(lm.x * w), int(lm.y * h)
                depth = depth_frame.get_distance(px, py)
                if depth == 0:
                    coords.append((0.0, 0.0, 0.0))
                else:
                    X, Y, Z = rs.rs2_deproject_pixel_to_point(depth_intrin, [px, py], depth)
                    coords.append((X, Y, Z))
        while len(coords) < expected_count:
            coords.append((0.0, 0.0, 0.0))
        return coords

    # Get all components
    landmarks_3d.extend(extract_landmarks(results.face_landmarks, 468))
    landmarks_3d.extend(extract_landmarks(results.pose_landmarks, 33))
    landmarks_3d.extend(extract_landmarks(results.left_hand_landmarks, 21))
    landmarks_3d.extend(extract_landmarks(results.right_hand_landmarks, 21))

    return np.array(landmarks_3d)


def capture_sign_3d():
    # Set up RealSense pipelines
    pipeline_color = rs.pipeline()
    pipeline_depth = rs.pipeline()

    config_color = rs.config()
    config_depth = rs.config()

    config_color.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config_depth.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    profile_color = pipeline_color.start(config_color)
    profile_depth = pipeline_depth.start(config_depth)

    depth_profile = profile_depth.get_stream(rs.stream.depth)
    depth_intrin = depth_profile.as_video_stream_profile().get_intrinsics()

    print("Capturing RGB+Depth with MediaPipe... Press 'q' or wait 8s to capture")

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    with mp_holistic.Holistic(static_image_mode=True, model_complexity=2) as holistic:
        start_time = time.time()
        while True:
            color_frames = pipeline_color.wait_for_frames()
            depth_frames = pipeline_depth.wait_for_frames()

            color_frame = color_frames.get_color_frame()
            depth_frame = depth_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            results = holistic.process(image_rgb)

            # Draw for preview
            annotated = color_image.copy()
            mp_drawing.draw_landmarks(
                annotated, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                annotated, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(
                annotated, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(
                annotated, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            cv2.imshow("MediaPipe Holistic", annotated)

            if (cv2.waitKey(5) & 0xFF == ord('q')) or (time.time() - start_time > 8):
                break

        landmarks_3d = get_3d_landmarks(results, color_image.shape, depth_frame, depth_intrin)

        pipeline_color.stop()
        pipeline_depth.stop()
        cv2.destroyAllWindows()

        # Save to parquet (543, 3)
        df = pd.DataFrame(landmarks_3d, columns=['x', 'y', 'z'])
        df['landmark_index'] = df.index
        df['frame'] = 0
        df.to_parquet("point_cloud.parquet", index=False)

        print("Captured and saved 3D landmark frame as point_cloud.parquet")


if __name__ == '__main__':
    capture_sign_3d()

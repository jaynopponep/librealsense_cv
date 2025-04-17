import cv2
import numpy as np
import pandas as pd
import time
from mmpose.apis import MMPoseInferencer

# four key points
POSE_KPTS = list(range(0, 17))
FACE_KPTS = list(range(17, 85))
LEFT_HAND_KPTS = list(range(85, 106))
RIGHT_HAND_KPTS = list(range(106, 127))

# MMPose inference model here ->
inferencer = MMPoseInferencer(
    pose2d='configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/vipnas_mbv3_coco_wholebody_256x192_dark.py',
    pose2d_weights='https://download.openmmlab.com/mmpose/top_down/vipnas/vipnas_mbv3_coco_wholebody_256x192_dark-e2158108_20211205.pth',
    det_model='human'
)

cap = cv2.VideoCapture(0)
start_time = time.time()
landmark_rows = []
frame_num = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame_num += 1

    frame = cv2.flip(frame, 1)
    frame_resized = cv2.resize(frame, (640, 480))
    image_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # perform the inferencing, collecting landmarks
    result_generator = inferencer(image_rgb)
    result = next(result_generator)

    if result and 'predictions' in result and result['predictions']:
        keypoints = result['predictions'][0]['keypoints']
        for i, (x, y, conf) in enumerate(keypoints):
            if conf < 0.05:
                continue
            h, w, _ = frame_resized.shape
            x_norm = x / w
            y_norm = y / h
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
            landmark_rows.append({
                'type': typ,
                'landmark_index': idx,
                'x': x_norm,
                'y': y_norm,
                'z': 0.0,
                'frame': frame_num
            })
            if conf > 0.5:
                cv2.circle(frame_resized, (int(x), int(y)), 3, (0, 255, 0), -1)

    cv2.imshow("MMPose 2D Capture", frame_resized)
    if cv2.waitKey(1) & 0xFF == ord('q') or (time.time() - start_time > 10):
        break

cap.release()
cv2.destroyAllWindows()

if landmark_rows:
    landmarks_df = pd.DataFrame(landmark_rows)
    landmarks_df.to_parquet("2d_landmarks.parquet", index=False)
    print("Saved 2D landmark data to 2d_landmarks.parquet")

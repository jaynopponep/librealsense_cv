import cv2
import openpifpaf
import numpy as np
import pandas as pd
import time

POSE_KPTS = list(range(0, 17))
FACE_KPTS = list(range(17, 85))
LEFT_HAND_KPTS = list(range(85, 106))
RIGHT_HAND_KPTS = list(range(106, 127))

predictor = openpifpaf.Predictor(checkpoint='resnet101-wholebody') # pre-trained model for whole body pose estimation

'''
model alteratives:
predictor = openpifpaf.Predictor(checkpoint='resnet50')
predictor = openpifpaf.Predictor(checkpoint='resnet101')
predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k30')
predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k16')
predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k16-wholebody')
predictor = openpifpaf.Predictor(checkpoint='resnet50-wholebody')
predictor = openpifpaf.Predictor(checkpoint='resnet101-wholebody')

predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k30-wholebody')

'''
cap = cv2.VideoCapture(0) #using the default camera (0) for capturing video, change the index for diff cameras
start_time = time.time() #start the timer for 10 seconds
landmark_rows = [] #list to store landmark data
frame_num = 0

while True:
    ret, frame = cap.read() #read a frame from the camera
    if not ret: #if frame not captured, continue to the next iteration
        continue
    frame_num += 1 #increment the frame number

    frame = cv2.flip(frame, 1) #flip the frame horizontally for better visualization
    # resize image to lower resolution to ease processing
    frame = cv2.resize(frame, (640, 480)) 
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #convert to RGB format for openpifpaf to be able to plot
    
    try: #predict keypoints using openpifpaf
        predictions, _, _ = predictor.numpy_image(image_rgb)
    except Exception as e: #catch any exception during prediction
        print("Error during predictor call:", e)
        continue

    if predictions:
        keypoints = predictions[0].data
        for i, (x, y, conf) in enumerate(keypoints):
            if conf < 0.05:
                continue
            h, w, _ = frame.shape
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
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
    
    cv2.imshow("OpenPifPaf 2D Capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or (time.time() - start_time > 10):
        break

cap.release()
cv2.destroyAllWindows()

if landmark_rows:
    landmarks_df = pd.DataFrame(landmark_rows)
    landmarks_df.to_parquet("2d_landmarks.parquet", index=False)
    print("Saved 2D landmark data to landmarks.parquet")
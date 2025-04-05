import cv2
import openpifpaf
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import time

def create_frame_landmarks(pred, frame, xyz):
    """
    Create a DataFrame of landmarks from an OpenPifPaf prediction.
    Assumes that `xyz` is a mapping DataFrame with at least the columns 'type' and 'landmark_index'
    for the pose landmarks.
    """
    if pred is None:
        # Create an empty dataframe if no prediction is available.
        landmarks = pd.DataFrame(columns=['type', 'landmark_index', 'x', 'y', 'z', 'frame'])
        return landmarks

    # Create a DataFrame from the prediction.
    # OpenPifPaf returns keypoints as an array of [x, y, score] values.
    pose = pd.DataFrame(pred.data, columns=['x', 'y', 'score'])
    pose['landmark_index'] = pose.index
    pose['type'] = 'pose'
    # OpenPifPaf does not output a z coordinate, so we set it to NaN.
    pose['z'] = np.nan

    # Assume that your mapping file (xyz) includes the keypoints for pose only.
    xyz_skel = (
        xyz[xyz['type'] == 'pose'][['type', 'landmark_index']]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    landmarks = pd.merge(xyz_skel, pose, on=['type', 'landmark_index'], how='left')
    landmarks['frame'] = frame
    return landmarks

def capture(xyz):
    """
    Capture video frames from the webcam, run OpenPifPaf inference,
    draw the detected pose on the image, and record landmarks.
    """
    all_landmarks = []
    cap = cv2.VideoCapture(0)
    # Initialize the OpenPifPaf predictor.
    predictor = openpifpaf.Predictor(checkpoint='resnet50')
    frame = 0
    start_time = time.time()

    while cap.isOpened():
        frame += 1
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a mirror view.
        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run OpenPifPaf inference.
        predictions, _, _ = predictor.numpy_image(rgb_image)
        # For simplicity, use only the first detected person (if any).
        pred = predictions[0] if predictions else None

        landmarks = create_frame_landmarks(pred, frame, xyz)
        all_landmarks.append(landmarks)

        # Draw keypoints and skeleton on the image if a prediction exists.
        if pred is not None:
            keypoints = pred.data  # shape (N, 3) with columns [x, y, score]
            # Draw each keypoint if confidence is above a threshold.
            for kp in keypoints:
                x, y, conf = kp
                if conf > 0.5:
                    cv2.circle(image, (int(x), int(y)), 3, (0, 255, 0), -1)

            # Define a skeleton (for the COCO keypoints order used by OpenPifPaf).
            skeleton = [
                (0, 1), (0, 2), (1, 3), (2, 4),
                (0, 5), (0, 6), (5, 7), (7, 9),
                (6, 8), (8, 10), (5, 6), (5, 11),
                (6, 12), (11, 12), (11, 13), (13, 15),
                (12, 14), (14, 16)
            ]
            for connection in skeleton:
                i, j = connection
                if i < len(keypoints) and j < len(keypoints):
                    x1, y1, conf1 = keypoints[i]
                    x2, y2, conf2 = keypoints[j]
                    if conf1 > 0.5 and conf2 > 0.5:
                        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        cv2.imshow('OpenPifPaf Pose Estimation', image)
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        elif time.time() - start_time > 8:
            break

    cap.release()
    cv2.destroyAllWindows()
    return all_landmarks

def capture_sign():
    BASE_DIR = 'data/asl-signs/'
    train = pd.read_csv(f'{BASE_DIR}/train.csv')
    # Load the mapping file that contains the landmark indices and types.
    xyz = pd.read_parquet(f'{BASE_DIR}/train_landmark_files/16069/695046.parquet')
    all_landmarks = capture(xyz)
    all_landmarks = pd.concat(all_landmarks).reset_index(drop=True)
    all_landmarks.to_parquet('landmarks.parquet')
    return

if __name__ == "__main__":
    capture_sign()
    print('Landmarks saved')

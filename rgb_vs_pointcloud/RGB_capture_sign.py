import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
import pandas as pd
import time
import os
import pyrealsense2 as rs
import numpy as np

def create_frame_landmarks(results,frame, xyz): # function to create a dataframe of landmarks for each frame

    xyz_skel=xyz[['type','landmark_index']].drop_duplicates().reset_index(drop=True).copy()
    
    face=pd.DataFrame()#creating 4dataframes for each of the landmarks
    pose=pd.DataFrame()
    left_hand=pd.DataFrame()
    right_hand=pd.DataFrame()
    if results.face_landmarks: #face landmarks
        for i,point in enumerate(results.face_landmarks.landmark):
            face.loc[i,['x','y','z']]=[point.x,point.y,point.z]
    if results.pose_landmarks: #pose landmarks
        for i,point in enumerate(results.pose_landmarks.landmark):
            pose.loc[i,['x','y','z']]=[point.x,point.y,point.z]
    if results.left_hand_landmarks: #left hand landmarks
        for i,point in enumerate(results.left_hand_landmarks.landmark):
            left_hand.loc[i,['x','y','z']]=[point.x,point.y,point.z]   
    if results.right_hand_landmarks: ##right hand landmarks
        for i,point in enumerate(results.right_hand_landmarks.landmark):
            right_hand.loc[i,['x','y','z']]=[point.x,point.y,point.z]
    face=face.reset_index().rename(columns={'index':'landmark_index'}).assign(type='face') #resetting the index and renaming the columns
    pose=pose.reset_index().rename(columns={'index':'landmark_index'}).assign(type='pose')
    left_hand=left_hand.reset_index().rename(columns={'index':'landmark_index'}).assign(type='left_hand')
    right_hand=right_hand.reset_index().rename(columns={'index':'landmark_index'}).assign(type='right_hand')
    landmarks=pd.concat([face,pose,left_hand,right_hand]).reset_index(drop=True) #concatenating the dataframes
    #print(landmarks.columns,xyz_skel.columns)
    landmarks=xyz_skel.merge(landmarks,how='left',on=['type','landmark_index']) #left merging the dataframes to get the x,y,z coordinates
    landmarks=landmarks.assign(frame=frame) ##assigning the frame number to the dataframe
    return landmarks


def capture(xyz): #opencv capture function to capture the video and landmarks
    all_landmarks=[]
    #realsense setup
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) 
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile  = pipeline.start(config) #start streaming depth + color
    align    = rs.align(rs.stream.color) #align depth to color stream

    #cv2 fallback (uncomment if you want to revert)
    #cap=cv2.VideoCapture(2)

    with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic: #using holistic model for pose detection
        frame=0
        start_time = time.time()
        while True:  # using RealSense, so we control the loop manually
            frame+=1
            frames      = pipeline.wait_for_frames()
            aligned     = align.process(frames)
            color_frame = aligned.get_color_frame()
            if not color_frame:
                continue

            #depth_frame = aligned.get_depth_f
            image = np.asanyarray(color_frame.get_data())
            image = cv2.flip(image, 1)

            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)         
            results = holistic.process(image_rgb) #process the image using holistic model

            landmarks=create_frame_landmarks(results,frame,xyz) #using our previously defined helper to create the landmarks dataframe
            all_landmarks.append(landmarks) #appending the landmarks to the list

            image.flags.writeable = True
            mp_drawing.draw_landmarks( #now we draw the landmarks on the image using mediapipe drawing utils
                image, 
                results.face_landmarks,  
                mp_holistic.FACEMESH_CONTOURS, #face landmarks
                landmark_drawing_spec=None, 
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
            cv2.imshow('MediaPipe Holistic (RealSense RGB)', image) #show the image with landmarks
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
            elif time.time() - start_time > 8:
                break   

        cv2.destroyAllWindows()
        pipeline.stop() 
    return all_landmarks
def capture_sign(): #function to store the landmarks in a parquet file
    BASE_DIR = os.path.join(os.getcwd(), 'data', 'asl-signs') #data directory
    train = pd.read_csv(f'{BASE_DIR}/train.csv')
    xyz=pd.read_parquet(f'{BASE_DIR}/train_landmark_files/16069/695046.parquet') #loading a sample landmark file to intialize the dataframe with the right shape
    all_landmarks=capture(xyz) #calling the capture function to get the landmarks
    all_landmarks=pd.concat(all_landmarks).reset_index(drop=True) ##concatenating the landmarks to make a single dataframe
    all_landmarks.to_parquet('2d_landmarks.parquet')
    return

if __name__ == "__main__": #main function to test these scripts in isolation
    capture_sign()
    print('Landmarks saved') #saved! 




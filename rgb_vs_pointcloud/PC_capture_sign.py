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
import open3d as o3d   # <-- added for point-cloud visualisation

#stream init
#realsense setup
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) 
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config) #start streaming depth + color
align   = rs.align(rs.stream.color) #align depth to color stream

# get intrinsics
color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
intrinsics   = color_stream.get_intrinsics()

def create_frame_landmarks(results,frame, xyz, depth_frame): # function to create a dataframe of landmarks for each frame

    xyz_skel=xyz[['type','landmark_index']].drop_duplicates().reset_index(drop=True).copy()
    
    face=pd.DataFrame()#creating 4dataframes for each of the landmarks
    pose=pd.DataFrame()
    left_hand=pd.DataFrame()
    right_hand=pd.DataFrame()

    w, h = intrinsics.width, intrinsics.height  # >>> NEW keep width/height handy

    # >>> NEW helper with bounds-check so we never call RealSense on invalid pixels
    def _real_z(px, py):
        if px < 0 or px >= w or py < 0 or py >= h:
            return np.nan                     # outside frame → give NaN
        d = depth_frame.get_distance(int(px), int(py))
        if d == 0:
            return np.nan                     # no depth → give NaN
        return rs.rs2_deproject_pixel_to_point(
            intrinsics, [float(px), float(py)], d
        )[2]                                  # we only need Z (meters)

    if results.face_landmarks: #face landmarks
        for i,point in enumerate(results.face_landmarks.landmark):
            px, py = point.x * w, point.y * h
            face.loc[i,['x','y','z']]=[point.x,point.y,_real_z(px,py)]
    if results.pose_landmarks: #pose landmarks
        for i,point in enumerate(results.pose_landmarks.landmark):
            px, py = point.x * w, point.y * h
            pose.loc[i,['x','y','z']]=[point.x,point.y,_real_z(px,py)]
    if results.left_hand_landmarks: #left hand landmarks
        for i,point in enumerate(results.left_hand_landmarks.landmark):
            px, py = point.x * w, point.y * h
            left_hand.loc[i,['x','y','z']]=[point.x,point.y,_real_z(px,py)]   
    if results.right_hand_landmarks: ##right hand landmarks
        for i,point in enumerate(results.right_hand_landmarks.landmark):
            px, py = point.x * w, point.y * h
            right_hand.loc[i,['x','y','z']]=[point.x,point.y,_real_z(px,py)]

    face=face.reset_index().rename(columns={'index':'landmark_index'}).assign(type='face') #resetting the index and renaming the columns
    pose=pose.reset_index().rename(columns={'index':'landmark_index'}).assign(type='pose')
    left_hand=left_hand.reset_index().rename(columns={'index':'landmark_index'}).assign(type='left_hand')
    right_hand=right_hand.reset_index().rename(columns={'index':'landmark_index'}).assign(type='right_hand')
    landmarks=pd.concat([face,pose,left_hand,right_hand]).reset_index(drop=True) #concatenating the dataframes
    landmarks=xyz_skel.merge(landmarks,how='left',on=['type','landmark_index']) #left merging the dataframes to get the x,y,z coordinates
    landmarks=landmarks.assign(frame=frame) ##assigning the frame number to the dataframe
    return landmarks


def capture(xyz): #opencv capture function to capture the video and landmarks
    all_landmarks=[]
    last_depth  = None   #will hold the most recent depth frame for point cloud
    #cap=cv2.VideoCapture(2)
    with mp_holistic.Holistic(min_detection_confidence=0.5,min_tracking_confidence=0.5) as holistic: #using holistic model for pose detection
        frame=0
        start_time = time.time()

        while True:
            frame += 1
            #pull synced frames
            frames      = pipeline.wait_for_frames()
            aligned     = align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            last_depth  = depth_frame          #save for point-cloud later

            color_image = np.asanyarray(color_frame.get_data())
            image       = cv2.flip(color_image, 1)

            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)         
            results = holistic.process(image_rgb) #process the image using holistic model

            landmarks=create_frame_landmarks(results,frame,xyz,depth_frame) #using our helper
            all_landmarks.append(landmarks) ##appending the landmarks to the list

            image.flags.writeable = True
            mp_drawing.draw_landmarks( #draw landmarks
                image, 
                results.face_landmarks,  
                mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None, 
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
            cv2.imshow('MediaPipe Holistic (RealSense depth-backed Z)', image) #show the image with landmarks
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
            elif time.time() - start_time > 8:
                break   

        #---- point-cloud generation just before shutdown ------------------
        if last_depth:
            pc      = rs.pointcloud()
            points  = pc.calculate(last_depth)
            verts   = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
            verts   = verts[verts[:, 2] > 0]   #drop invalid
            #basic crop around camera centre (customise to taste)
            x_min, x_max = -0.5,  0.5
            y_min, y_max = -1.2,  0.2
            z_min, z_max =  0.2,  1.5
            mask = (
                (verts[:,0] > x_min)&(verts[:,0] < x_max)&
                (verts[:,1] > y_min)&(verts[:,1] < y_max)&
                (verts[:,2] > z_min)&(verts[:,2] < z_max)
            )
            filt = verts[mask]
            if filt.shape[0]:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(filt)
                o3d.visualization.draw_geometries(
                    [pcd],
                    window_name='Captured PointCloud'
                )
            else:
                print("No valid points in range for point-cloud.")
        #--------------------------------------------------------------------

        pipeline.stop()
        cv2.destroyAllWindows()
    return all_landmarks


def capture_sign(): #function to store the landmarks in a parquet file
    BASE_DIR = os.path.join(os.getcwd(), 'data', 'asl-signs') #data directory
    train = pd.read_csv(f'{BASE_DIR}/train.csv')
    xyz=pd.read_parquet(f'{BASE_DIR}/train_landmark_files/16069/695046.parquet') #loading a sample landmark file to intialize the dataframe with the right shape
    all_landmarks=capture(xyz) #calling the capture function to get the landmarks
    all_landmarks=pd.concat(all_landmarks).reset_index(drop=True) ##concatenating the landmarks to make a single dataframe
    all_landmarks.to_parquet('3d_andmarks.parquet')
    return

if __name__ == "__main__": #main function to test these scripts in isolation
    capture_sign()
    print('Landmarks saved') #saved! 

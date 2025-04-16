import pandas as pd #dependncies
import numpy as np
import pyarrow.parquet as pq
import tensorflow as tf 
import os

def get_distance(df, a_type, a_index, b_type, b_index): #distance function to calculate the distance between two points in 3D space
    def try_get_point(df, typ, ids): # get a point from the dataframe based on type and ids
        for i in ids:
            point = df[(df['type'] == typ) & (df['landmark_index'] == i)][['x', 'y', 'z']].values
            if point.size > 0 and point[0][2] > 0:
                return point[0] #return the point if it exists and z coordinate is greater than 0
        return None

    hand_ids = [8, 7, 6, 5, 9] if a_type == 'right_hand' and a_index == 8 else [a_index] #right hand ids
    face_ids = [1, 60, 61, 46, 48, 0] if b_type == 'face' and b_index == 1 else [b_index] #face ids

    a = try_get_point(df, a_type, hand_ids) #get the point for a_type
    b = try_get_point(df, b_type, face_ids) #get the point for b_type

    if a is None or b is None: #if either point is None, return 1.0 as a fallback distance. aka the points are too far apart
        return 1.0  # fallback

    return np.linalg.norm(a - b) #using vector norm between a and b



    
def classifier(): #function to classify the sign language gesture based on the point cloud data
    BASE_DIR = os.path.join('data', 'asl-signs')  # data directory containing train.csv
    train_csv_path = os.path.join(BASE_DIR, 'train.csv')
    train_df = pd.read_csv(train_csv_path)
    
    # Create mappings from sign names to ordinal labels and back
    train_df['sign_ord'] = train_df['sign'].astype('category').cat.codes 
    SIGN2ORD = dict(zip(train_df['sign'], train_df['sign_ord'])) #convert sign names to ordinal labels
    ORD2SIGN = dict(zip(train_df['sign_ord'], train_df['sign'])) #convert ordinal labels back to sign names
    
    ROWS_PER_FRAME = 543  # number of landmarks expected per frame
    
    def load_relevant_data_subset(pq_path): 
        """Load labeled point cloud data and prioritize pose, hands, and face landmarks."""
        ROWS_PER_FRAME = 543
        try:
            data = pd.read_parquet(pq_path, columns=['frame', 'x', 'y', 'z', 'type', 'landmark_index']) #try to load the parquet file with the specified columns
        except Exception: #err
            # Fallback for legacy files
            data = pd.read_parquet(pq_path) #else load the parquet file without specifying columns

        # Keep only relevant landmark types
        data = data[data['type'].isin(['pose', 'left_hand', 'right_hand', 'face'])] ##filter the data to keep only relevant landmark types

        # Consistent ordering: first by type, then landmark_index
        type_order = {'pose': 0, 'left_hand': 1, 'right_hand': 2, 'face': 3} # mapping of landmark types to order
        data['type_order'] = data['type'].map(type_order) #map the type to the order
        data = data.sort_values(['frame', 'type_order', 'landmark_index'])# #sort the data by frame, type_order and landmark_index

        frames_array = []
        if 'frame' in data.columns: #check if frame column exists in the data
            for _, frame_data in data.groupby('frame'): ##group the data by frame
                coords = frame_data[['x', 'y', 'z']].to_numpy()# #get the coordinates of the landmarks in numpy array format
                if coords.shape[0] < ROWS_PER_FRAME: ##check if the number of coordinates is less than the expected number of rows per frame
                    coords = np.vstack([coords, np.zeros((ROWS_PER_FRAME - coords.shape[0], 3))]) # #pad the coordinates with zeros to make it the expected number of rows per frame
                elif coords.shape[0] > ROWS_PER_FRAME: #else if the number of coordinates is greater than the expected number of rows per frame
                    coords = coords[:ROWS_PER_FRAME] #truncate the coordinates to make it the expected number of rows per frame
                frames_array.append(coords)
        else: # #if frame column does not exist, process the data as a single frame
            coords = data[['x', 'y', 'z']].to_numpy()# repeat the same steps from before
            if coords.shape[0] < ROWS_PER_FRAME:
                coords = np.vstack([coords, np.zeros((ROWS_PER_FRAME - coords.shape[0], 3))])
            elif coords.shape[0] > ROWS_PER_FRAME:
                coords = coords[:ROWS_PER_FRAME]
            right_hand_tip_to_nose = get_distance(frame_data, 'right_hand', 8, 'face', 1)  #calculate the distance between right hand tip and nose
            extra_feature = np.full((coords.shape[0], 1), right_hand_tip_to_nose)    #create an extra feature based on the distance calculated above
            coords = np.hstack([coords, extra_feature]) #append the extra feature to the coordinates usin hstack
            frames_array.append(coords) #append the coordinates to the frames array

        return np.array(frames_array, dtype=np.float32) ##convert the frames array to numpy array of type float32

    
    # Load the point cloud data and prepare input tensor
    pq_path = 'labeled_point_cloud.parquet' #saved point cloud data as parquet file
    frames = load_relevant_data_subset(pq_path)
    # Ensure the shape is (1, 543, 3) for a single frame
    # (If multiple frames were present, frames.shape[0] would be >1 accordingly.)
    
    # Initialize TFLite interpreter and get the prediction function
    model_path = "model.tflite"
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    prediction_fn = interpreter.get_signature_runner("serving_default")
    
    # Run inference
    output = prediction_fn(inputs=frames)
    class_idx = int(np.argmax(output["outputs"]))
    predicted_sign = ORD2SIGN.get(class_idx, "<UNK>")
    
    return predicted_sign

# If executed as a script, print the result
if __name__ == "__main__":
    print(classifier())
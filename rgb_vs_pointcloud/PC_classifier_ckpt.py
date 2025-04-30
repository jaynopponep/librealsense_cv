def pc_classifier():
    import pandas as pd 
    import numpy as np
    import pyarrow.parquet as pq
    import tensorflow.compat.v1 as tf  
    import os
    import sys

    tf.disable_eager_execution()
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(BASE_DIR)
    sys.path.append(os.path.join(BASE_DIR, 'models'))
    sys.path.append(os.path.join(BASE_DIR, 'utils'))
    
    try:
        import pointnet_cls as MODEL
    except ImportError:
        try:
            sys.path.append('models')
            import pointnet_cls as MODEL
        except ImportError:
            print("Could not import pointnet_cls model. Make sure it exists in the models directory.")
            return "ERROR: Model import failed"

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
    

    BASE_DIR = os.path.join('data', 'asl-signs')  # data directory containing train.csv
    train_csv_path = os.path.join(BASE_DIR, 'train.csv')
    train_df = pd.read_csv(train_csv_path)
    
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
            data = pd.read_parquet(pq_path) #else load the parquet file without specifying columns

        data = data[data['type'].isin(['pose', 'left_hand', 'right_hand', 'face'])] ##filter the data to keep only relevant landmark types

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
            # Calculate additional features if needed
            try:
                right_hand_tip_to_nose = get_distance(data, 'right_hand', 8, 'face', 1)  #calculate the distance between right hand tip and nose
                extra_feature = np.full((coords.shape[0], 1), right_hand_tip_to_nose)    #create an extra feature based on the distance calculated above
                coords = np.hstack([coords, extra_feature]) #append the extra feature to the coordinates usin hstack
            except Exception as e:
                # If feature calculation fails, just use the coordinates
                pass
            frames_array.append(coords) #append the coordinates to the frames array

        return np.array(frames_array, dtype=np.float32) ##convert the frames array to numpy array of type float32

    
    pq_path = '3d_landmarks.parquet' #saved point cloud data as parquet file
    frames = load_relevant_data_subset(pq_path)
    
    num_points = 1024  
    if frames.shape[1] > num_points:
        indices = np.random.choice(frames.shape[1], num_points, replace=False)
        point_cloud = frames[0, indices, :3]  
    else:
        indices = np.random.choice(frames.shape[1], num_points, replace=True)
        point_cloud = frames[0, indices, :3] 
    
    point_cloud = np.reshape(point_cloud, (1, num_points, 3))
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    
    checkpoint_path = os.path.join('log', 'train', 'model.ckpt')
    
    with tf.Graph().as_default():
        with tf.Session(config=config) as sess:
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(1, num_points)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            MODEL.NUM_CLASSES = 2  
            pred, _ = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=None)
            
            saver = tf.train.Saver()
            
            saver.restore(sess, checkpoint_path)
            
            feed_dict = {
                pointclouds_pl: point_cloud,
                is_training_pl: False
            }
            
            logits = sess.run(pred, feed_dict=feed_dict)
            
            class_idx = np.argmax(logits[0])
            
    predicted_sign = ORD2SIGN.get(class_idx, "<UNK>")
    
    return predicted_sign

if __name__ == "__main__":
    print(pc_classifier())

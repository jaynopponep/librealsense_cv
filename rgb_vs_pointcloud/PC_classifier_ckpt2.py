def pc_classifier():
    """Function to classify based on point cloud data using TensorFlow checkpoint"""
    import pandas as pd
    import numpy as np
    import pyarrow.parquet as pq
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    import os
    import sys

    # Disable eager execution for TF1.x compatibility
    tf.disable_eager_execution()

    # Define PointNet model directly in this file to avoid import issues
    def placeholder_inputs(batch_size, num_point):
        """Return the placeholder for the dataset"""
        pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
        labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
        return pointclouds_pl, labels_pl

    # Modify get_model to handle BatchNormalization correctly in graph mode
    @tf.function
    def get_model(point_cloud, is_training, bn_decay=None):
        batch_size = tf.shape(point_cloud)[0]  # ← CHANGED from point_cloud.shape[0]
        num_point = tf.shape(point_cloud)[1]  # ← CHANGED from point_cloud.shape[1]
        end_points = {}

        # Input transformer
        input_image = tf.expand_dims(point_cloud, -1)

        # Point functions
        net = tf.keras.layers.Conv2D(64, [1, 3], padding='valid', activation=tf.nn.relu, name='conv1')(input_image)
        net = tf.keras.layers.BatchNormalization(name='bn1')(net, training=is_training)

        net = tf.keras.layers.Conv2D(64, [1, 1], padding='valid', activation=tf.nn.relu, name='conv2')(net)
        net = tf.keras.layers.BatchNormalization(name='bn2')(net, training=is_training)

        net = tf.keras.layers.Conv2D(64, [1, 1], padding='valid', activation=tf.nn.relu, name='conv3')(net)
        net = tf.keras.layers.BatchNormalization(name='bn3')(net, training=is_training)

        net = tf.keras.layers.Conv2D(128, [1, 1], padding='valid', activation=tf.nn.relu, name='conv4')(net)
        net = tf.keras.layers.BatchNormalization(name='bn4')(net, training=is_training)

        net = tf.keras.layers.Conv2D(1024, [1, 1], padding='valid', activation=tf.nn.relu, name='conv5')(net)
        net = tf.keras.layers.BatchNormalization(name='bn5')(net, training=is_training)

        # Symmetric function: max pooling
        net = tf.reduce_max(net, axis=1, keepdims=True)

        # MLP for classification
        net = tf.reshape(net, [batch_size, -1])
        net = tf.keras.layers.Dense(512, activation=tf.nn.relu, name='fc1')(net)
        net = tf.keras.layers.BatchNormalization(name='bn6')(net, training=is_training)
        net = tf.keras.layers.Dropout(rate=0.3, name='dp1')(net, training=is_training)

        net = tf.keras.layers.Dense(256, activation=tf.nn.relu, name='fc2')(net)
        net = tf.keras.layers.BatchNormalization(name='bn7')(net, training=is_training)
        net = tf.keras.layers.Dropout(rate=0.3, name='dp2')(net, training=is_training)

        # The last FC layer outputs logits with NUM_CLASSES nodes
        net = tf.keras.layers.Dense(NUM_CLASSES, activation=None, name='fc3')(net)

        return net, end_points

    def get_distance(df, a_type, a_index, b_type, b_index):
        """Calculate distance between two points in 3D space"""
        def try_get_point(df, typ, ids):
            for i in ids:
                point = df[(df['type'] == typ) & (df['landmark_index'] == i)][['x', 'y', 'z']].values
                if point.size > 0 and point[0][2] > 0:
                    return point[0]
            return None

        hand_ids = [8, 7, 6, 5, 9] if a_type == 'right_hand' and a_index == 8 else [a_index]
        face_ids = [1, 60, 61, 46, 48, 0] if b_type == 'face' and b_index == 1 else [b_index]

        a = try_get_point(df, a_type, hand_ids)
        b = try_get_point(df, b_type, face_ids)

        if a is None or b is None:
            return 1.0  # fallback

        return np.linalg.norm(a - b)

    # Set up labels for your model
    BASE_DIR = os.path.join('data', 'asl-signs')
    try:
        train_csv_path = os.path.join(BASE_DIR, 'train.csv')
        train_df = pd.read_csv(train_csv_path)

        train_df['sign_ord'] = train_df['sign'].astype('category').cat.codes
        SIGN2ORD = dict(zip(train_df['sign'], train_df['sign_ord']))
        ORD2SIGN = dict(zip(train_df['sign_ord'], train_df['sign']))
    except Exception as e:
        print(f"Warning: Could not load class mappings: {e}")
        ORD2SIGN = {0: "class_0", 1: "class_1"}

    ROWS_PER_FRAME = 543

    def load_relevant_data_subset(pq_path):
        try:
            data = pd.read_parquet(pq_path, columns=['frame', 'x', 'y', 'z', 'type', 'landmark_index'])
        except Exception:
            data = pd.read_parquet(pq_path)

        data = data[data['type'].isin(['pose', 'left_hand', 'right_hand', 'face'])]

        type_order = {'pose': 0, 'left_hand': 1, 'right_hand': 2, 'face': 3}
        data['type_order'] = data['type'].map(type_order)
        data = data.sort_values(['frame', 'type_order', 'landmark_index'])

        frames_array = []
        if 'frame' in data.columns:
            for _, frame_data in data.groupby('frame'):
                coords = frame_data[['x', 'y', 'z']].to_numpy()
                if coords.shape[0] < ROWS_PER_FRAME:
                    coords = np.vstack([coords, np.zeros((ROWS_PER_FRAME - coords.shape[0], 3))])
                elif coords.shape[0] > ROWS_PER_FRAME:
                    coords = coords[:ROWS_PER_FRAME]
                frames_array.append(coords)
        else:
            coords = data[['x', 'y', 'z']].to_numpy()
            if coords.shape[0] < ROWS_PER_FRAME:
                coords = np.vstack([coords, np.zeros((ROWS_PER_FRAME - coords.shape[0], 3))])
            elif coords.shape[0] > ROWS_PER_FRAME:
                coords = coords[:ROWS_PER_FRAME]
            frames_array.append(coords)

        return np.array(frames_array, dtype=np.float32)

    # Ensure we have the checkpoint path
    pointnet_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(pointnet_dir, 'log', 'model.ckpt')
    if not os.path.exists(checkpoint_path + '.meta'):
        possible_paths = [
            os.path.join('log', 'model.ckpt'),
            os.path.join(os.path.dirname(pointnet_dir), 'log', 'model.ckpt'),
            os.path.join(os.path.dirname(os.path.dirname(pointnet_dir)), 'log', 'model.ckpt')
        ]
        found = False
        for path in possible_paths:
            if os.path.exists(path + '.meta'):
                checkpoint_path = path
                found = True
                print(f"Found checkpoint at: {checkpoint_path}")
                break
        if not found:
            raise FileNotFoundError(f"Could not find checkpoint files at {checkpoint_path} or any standard locations")

    # Load the point cloud data
    pq_path = '3d_landmarks.parquet'
    frames = load_relevant_data_subset(pq_path)

    # Set number of classes to match your trained model
    global NUM_CLASSES
    NUM_CLASSES = 2  # Change this to match your model

    # Adapt the frames array to PointNet input format
    num_points = 1024
    if frames.shape[1] > num_points:
        indices = np.random.choice(frames.shape[1], num_points, replace=False)
        point_cloud = frames[0, indices, :3]
    else:
        indices = np.random.choice(frames.shape[1], num_points, replace=True)
        point_cloud = frames[0, indices, :3]

    point_cloud = np.reshape(point_cloud, (1, num_points, 3))

    # Set up TensorFlow session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False

    with tf.Graph().as_default():
        with tf.Session(config=config) as sess:
            pointclouds_pl, labels_pl = placeholder_inputs(1, num_points)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            pred, _ = get_model(pointclouds_pl, is_training_pl, bn_decay=None)
            saver = tf.train.Saver()

            sess.run(tf.global_variables_initializer())
            try:
                saver.restore(sess, checkpoint_path)
                print("Model restored from:", checkpoint_path)
            except Exception as e:
                print(f"Error restoring model: {e}")
                return "ERROR: Model restore failed"

            feed_dict = {
                pointclouds_pl: point_cloud,
                is_training_pl: False
            }

            try:
                logits = sess.run(pred, feed_dict=feed_dict)
                class_idx = np.argmax(logits[0])
                predicted_sign = ORD2SIGN.get(class_idx, f"Unknown Class {class_idx}")
                return predicted_sign
            except Exception as e:
                print(f"Error during inference: {e}")
                return "ERROR: Inference failed"

if __name__ == "__main__":
    print(pc_classifier())

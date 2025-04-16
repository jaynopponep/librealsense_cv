import pandas as pd
import numpy as np
import tensorflow as tf
import os

def get_distance(df, a_type, a_index, b_type, b_index):
    def try_get_point(df, typ, ids):
        for i in ids:
            point = df[(df['type'] == typ) & (df['landmark_index'] == i)][['x', 'y', 'z']].values
            if point.size > 0:
                return point[0]
        return None

    hand_ids = [8, 7, 6, 5, 9] if a_type == 'right_hand' and a_index == 8 else [a_index]
    face_ids = [1, 60, 61, 46, 48, 0] if b_type == 'face' and b_index == 1 else [b_index]
    a = try_get_point(df, a_type, hand_ids)
    b = try_get_point(df, b_type, face_ids)
    if a is None or b is None:
        return 1.0
    return np.linalg.norm(a - b)

def classifier():
    BASE_DIR = os.path.join('data', 'asl-signs')
    train_csv_path = os.path.join(BASE_DIR, 'train.csv')
    train_df = pd.read_csv(train_csv_path)
    train_df['sign_ord'] = train_df['sign'].astype('category').cat.codes
    SIGN2ORD = dict(zip(train_df['sign'], train_df['sign_ord']))
    ORD2SIGN = dict(zip(train_df['sign_ord'], train_df['sign']))
    
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
            for frame_num, frame_data in data.groupby('frame'):
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
    
    pq_path = "2d_landmarks.parquet"
    frames = load_relevant_data_subset(pq_path)
    
    model_path = "model.tflite"
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    prediction_fn = interpreter.get_signature_runner("serving_default")
    
    output = prediction_fn(inputs=frames)
    class_idx = int(np.argmax(output["outputs"]))
    predicted_sign = ORD2SIGN.get(class_idx, "<UNK>")
    return predicted_sign

if __name__ == "__main__":
    print(classifier())

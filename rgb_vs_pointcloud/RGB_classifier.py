import pandas as pd #dependncies
import numpy as np
import tensorflow as tf 
import os 
'''
IMPORTANT NOTICE:
this code does NOT run on the same python version as the pointcloud capture. intel realsense requires that the python version be
3.7 or lower, but mediapipe only works on 3.9+. the rgb capture/classification uses python 3.12 to take advantage of mediapipe holistic to get z coordinates.
the pointcloud capture/classification uses python 3.7 to take advantage of the intel realsense camera and open3d for point cloud processing.
the goal of the project is to see the comparison of pointcloud data vs mediapipe's z coordinates for sign language classification.

'''
def classifier():
    BASE_DIR = os.path.join(os.getcwd(), 'data', 'asl-signs') #data directory
    train = pd.read_csv(f'{BASE_DIR}/train.csv')


    def run_model(model_path, frames): #classifier itself
        # Initialize the TensorFlow Lite interpreter
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        # Get the list of available signatures
        found_signatures = list(interpreter.get_signature_list().keys())
        REQUIRED_SIGNATURE = "serving_default"

        # Check if the required signature is available
        if REQUIRED_SIGNATURE not in found_signatures:
            raise Exception('Required input signature not found.')

        # Get the prediction function from the interpreter
        prediction_fn = interpreter.get_signature_runner(REQUIRED_SIGNATURE)

        # Run the prediction function with the input frames
        output = prediction_fn(inputs=frames)
        sign = np.argmax(output["outputs"])

        return sign



    train['sign_ord'] = train['sign'].astype('category').cat.codes #helper functions to convert sign to ordinal and vice versa
    SIGN2ORD = train[['sign', 'sign_ord']].set_index('sign').squeeze().to_dict()
    ORD2SIGN = train[['sign_ord', 'sign']].set_index('sign_ord').squeeze().to_dict()

    ROWS_PER_FRAME = 543  

    def load_relevant_data_subset(pq_path): #load data in proper format
        data_columns = ['x', 'y', 'z']
        data = pd.read_parquet(pq_path, columns=data_columns)
        #print(data.columns)
        n_frames = int(len(data) / ROWS_PER_FRAME)
        data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
        return data.astype(np.float32)

    model_path = os.path.join(os.getcwd(),"model.tflite") #model path
    pq_path = os.path.join(os.getcwd(),'2d_landmarks.parquet') #landmark path, this is the file that will be used to predict the sign
    frames = load_relevant_data_subset(pq_path)
    #print(frames.shape)
    sign = ORD2SIGN[run_model(model_path, frames)]
    #print(f"Predicted sign: {sign}")
    return sign

if __name__ == "__main__":
    print(classifier())
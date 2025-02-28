import zmq
import numpy as np
import cv2

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:5555")
socket.setsockopt(zmq.SUBSCRIBE, b"")

width, height = 640, 480 

while True:
    frame_bytes = socket.recv()
    frame_np = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((height, width, 3))

    cv2.imshow("Received Frame", frame_np)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()


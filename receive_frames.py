import zmq
import numpy as np
import cv2
from ultralytics import YOLO
import mediapipe as mp

model = YOLO("models/best.pt")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.4, min_tracking_confidence=0.4)

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'are',
    27: 'hello', 28: 'how', 29: 'iloveyou', 30: 'what-', 31: 'you'
}

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:5555")  
socket.setsockopt_string(zmq.SUBSCRIBE, "")

width, height = 640, 480

print("Listening for frames... Press 'q' to exit.")
while True:
    try:
        while socket.poll(1, zmq.POLLIN):  
            frame_bytes = socket.recv(zmq.DONTWAIT)  

        frame_np = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((height, width, 3)).copy()

        frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        predictions = model.predict(frame_np, conf=0.5)

        boxes = predictions[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])  
            conf = float(box.conf[0])  

            label = labels_dict.get(cls, f"Class {cls}")

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                x_vals = [landmark.x * frame_np.shape[1] for landmark in hand_landmarks.landmark]
                y_vals = [landmark.y * frame_np.shape[0] for landmark in hand_landmarks.landmark]
                x_min, x_max = int(min(x_vals)), int(max(x_vals))
                y_min, y_max = int(min(y_vals)), int(max(y_vals))

                pad = 40
                x1, y1, x2, y2 = max(x_min - pad, 0), max(y_min - pad, 0), min(x_max + pad, width), min(y_max + pad, height)

            cv2.rectangle(frame_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_np, f"{label} ({conf:.2f})", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame_np, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        cv2.imshow("YOLOv8 Hand Sign Detection", frame_np)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except zmq.Again:
        continue
    except Exception as e:
        print(f"Error processing frame: {e}")
        continue

cv2.destroyAllWindows()

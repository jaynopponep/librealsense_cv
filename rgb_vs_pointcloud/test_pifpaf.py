import time
import cv2
import numpy as np
import openpifpaf
from openpifpaf import show, Annotation
import torch
torch.backends.cudnn.benchmark = True


'''
this is a debug script to see the full capabilities of the shufflenet2k30 model on live video feed.
it will show the full body pose, face, left hand and right hand keypoints on the video feed.
mainly we want the face and hands to be detected, but the full body pose is also shown.
while this is not the best model for realtime inference, it is unfortunately one of the few models that can run on python 3.7.
'''
#load the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k16-wholebody')
painter   = show.AnnotationPainter()

#opencv stuff
cap   = cv2.VideoCapture(0)
start = time.time()

while True:
    ret, frame_bgr = cap.read()
    if not ret:
        break

    #prerocess the frame
    frame_bgr = cv2.flip(frame_bgr, 1)
    #frame_bgr = cv2.resize(frame_bgr, (640, 480))
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_idx = frame_bgr.shape[0] * frame_bgr.shape[1] #get the frame idk

    try:
        
        if frame_idx % 2 == 0:
            preds, _, _ = predictor.numpy_image(frame_rgb)
    
    except Exception as e:
        print("PifPaf error:", e)
        continue

    if preds:
        filtered = []
        
        #matplotlib stuff, ignore for now
        with show.Canvas.image(frame_rgb) as ax:
            painter.annotations(ax, preds)
            ax.axis('off')

            # Grab the RGBA buffer from the figure
            fig = ax.figure
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8) # get the buffer from the canvas
            overlay = buf.reshape(h, w, 3) # reshape the buffer to the correct shape
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR) # convert the overlay to BGR format for opencv display
        disp = cv2.resize(overlay_bgr, (frame_bgr.shape[1], frame_bgr.shape[0]))# # resize the overlay to the same size as the original frame
    else:
        disp = frame_bgr # if no predictions, just show the original frame

    #show to user the landmarks predicted
    cv2.imshow("Live OpenPifPaf (ShuffleNetV2 WholeBody)", disp)

    # Quit on 'q' or after 15s
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if time.time() - start > 30: #timer for 30 s
        break

cap.release()
cv2.destroyAllWindows()

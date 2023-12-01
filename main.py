#--------------------------------------------------------------------------------------------------------------
# imports 
#--------------------------------------------------------------------------------------------------------------
import json
import cv2
import torch
import time
import numpy as np 
import tensorflow as tf
from sklearn.preprocessing import normalize
import os
import sys
#yolo
from ultralytics import YOLO 
# retinaface 
from retinaface import RetinaFace
#bytetrack
from nets import nn
from utils import util
# our custom functions to simplify the code
from custom_functions.face_embedding import *
from custom_functions.annotate import *

#--------------------------------------------------------------------------------------------------------------

print("Reading configurations...")
# read configuration
json_file_path = 'configurations/config.json'
# read from a json file
with open(json_file_path, 'r') as json_file:
    configuration = json.load(json_file)

#--------------------------------------------------------------------------------------------------------------

# set the device to run the computation
## for torch
if torch.cuda.is_available():
    devicetorun = "cuda"
else:
    devicetorun = "cpu"
print(f'Torch models are using {devicetorun}')

## for tensorflow
print(f"Num GPUs Available for Tensorflow models: {len(tf.config.experimental.list_physical_devices('GPU'))}")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Limit GPU memory growth (optional)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Set visible GPUs
    tf.config.experimental.set_visible_devices(gpus, 'GPU')

#--------------------------------------------------------------------------------------------------------------
# initialize models
print("Initializing models...")
yolov8model = YOLO(configuration['yolov8_model_path']) #yolov8
face_model = tf.keras.models.load_model(configuration['ghostfacenets_model_path'], compile=False)

#--------------------------------------------------------------------------------------------------------------

# read and embed query 
print("Embedding queries...")
query_embeddings = generate_query_embeddings(configuration['query_path'], face_model)

#--------------------------------------------------------------------------------------------------------------

# initialize video reader and mot model (bytetrack)
cap = reader = cv2.VideoCapture(configuration['video_path'])
cv2.namedWindow("Output Video", cv2.WINDOW_NORMAL) 
fps = int(cap.get(cv2.CAP_PROP_FPS))
bytetrack = nn.BYTETracker(fps)

#--------------------------------------------------------------------------------------------------------------
intruder_id = None
previous_frame = None
print("--->>ENTERING MAIN LOOP<<---")
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break


    if intruder_id==None:
        # head detection using Yolov8 model
        head_detections = yolov8model(frame, conf=configuration['yolov8_confidence'], device=devicetorun)
        previous_frame = frame
        for result in head_detections:
                boxes = []
                confidences = []
                object_classes = []
                detections = []
                for r in result.boxes.data.tolist():
                    xd1, yd1, xd2, yd2, score, class_id = r
                    xd1, xd2, yd1, yd2 = int(xd1),int(xd2),int(yd1), int(yd2)
                    class_id = int(class_id)
                    boxes.append([xd1, yd1, xd2, yd2])
                    confidences.append(score)
                    object_classes.append(0)
                    detections.append([xd1, yd1, xd2, yd2, score])

        outputs = bytetrack.update( np.array(boxes),
                                    np.array(confidences),
                                    np.array(object_classes))


        if len(outputs) > 0:
            boxes = outputs[:, :4]
            identities = outputs[:, 4]
            object_classes = outputs[:, 6]
            print('IDENTITIES DATATYPE',type(identities))
            intruder_id = crop_and_find(frame, boxes, identities, face_model, query_embeddings)
            if intruder_id!=None:
                # RUN SOT
                break

            for i, box in enumerate(boxes):
                if object_classes[i] != 0:  # 0 is for head class (Custom Yolov8 model)
                    continue
                x1, y1, x2, y2 = list(map(int, box))
                # get ID of object
                index = int(identities[i]) if identities is not None else 0
                
                draw_annotation_mot(frame, x1, y1, x2, y2, index) # annotate based on the 

            # pass to retinaface and then do face recognition
            # crop from frame the boxes and identities
            
    else:
        # RUN SOT
        # lst = list(lst)
        # index = lst.index(42)
        # sot_bbox = run_sot(frame or frames, bbox from the MOT ) and return bbox of the intruder in the next frame
        # previous_frame, frame, 
        pass
       


    cv2.imshow('Output Video', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
#--------------------------------------------------------------------------------------------------------------
print("--->>EXITING MAIN LOOP<<---")


#--------------------------------------------------------------------------------------------------------------



print('process complete')
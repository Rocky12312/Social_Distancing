#Import the necessary required packages
import os
import cv2
import imutils
import argparse
import numpy as np

#Importing the detection function(and its configuration) 
import people_detection_using_yolo
#Using scipy distance for computing distance matrix from a collection of raw observation vectors stored in an array
from scipy.spatial import distance as dist


#Loading the COCO class labels on which the Yolo(you only look once) model was trained on
#Link to pretrained model weights( https://www.kaggle.com/aruchomu/data-for-yolo-v3-kernel ) ,visit this link and grab the yolov3 weights and coco.names files
MODEL_PATH = "Yolo_pretrained_files"
labelsPath = os.path.sep.join([MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

"""Coco class labels ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus',
       'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
       'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
       'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
       'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
       'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
       'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
       'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
       'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
       'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
       'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor',
       'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
       'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
       'scissors', 'teddy bear', 'hair drier', 'toothbrush']  """
       
#We will be only using person from this list
#Path to Yolo weights and model
#config which we created
weightsPath = os.path.sep.join([MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([MODEL_PATH, "yolov3.cfg"])

#Loading the Yolo object detector trained on COCO dataset which consist of 80 classes(Now one thing here is that as the Yolo is trained on 80 classes from Coco so it will detect 80 classes which--
#Which inlude classes which are are of no use for our social distancing detection task so we will modify our algo to detect only person in video frames.
print("loading the Yolo model from disk")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

#Checking if we are going to use GPU
USE_GPU = False
if USE_GPU:
    #Set CUDA as the preferable backend and target
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

#Determinimg the output layer names that are going to need from Yolo('yolo_82', 'yolo_94', 'yolo_106'--these are the layers always good to use pretrained models as they have been trained on -- very #huge data)
layernames_needed = net.getLayerNames()
layernames_needed = [layernames_needed[i[0] - 1] for i in net.getUnconnectedOutLayers()]


#Capturing the frames(using web camera(use 0 in cv2.VideoCapture(0) or use raspberry pi camera) or if want to use some video files for the task just provide the video file path at place of 0 in --
#next line(0 for capturing the live streams)
cap = cv2.VideoCapture(0)

#Check if camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video stream or file")
    
#Default resolutions of the frame are obtained.The default resolutions are system dependent.
#We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

#Defining the minimum safe distance(in pixels) that two people can be from each other
MIN_DISTANCE = 50


while True:
    # read the next frame from the file
    (ret, frame) = cap.read()

    #If the capture function do not return an output so we can break from the loop otherwise frames are captured and will continue with the process of people's detection
    if not ret:
        break

    #Resize the frame and then detect people (and only people) in it
    frame = imutils.resize(frame, width=700)
    #Will be used for extracting the centroid of peoples
    results = people_detection_using_yolo.detect_people(frame, net, layernames_needed, personIdx=LABELS.index("person"))

    #Initialize a set of indexes for the people which violate the minimum social distancing criteria
    #Using a set so that a single person may not included in people_index_violation multiple times
    people_index_violation = set()
    Total_number_of_people = set()

    #Make sure before marking an index under social distancing violation that there are at least two people detected within that proximity
    if len(results) >= 2:
        #Extract all centroids from the results and compute the euclidean distances between all pairs of the centroids
        people_centroids = np.array([r[2] for r in results])
        #Computing the euclidean distance between all pair of centroids
        D = dist.cdist(people_centroids, people_centroids, metric="euclidean")

        # loop over the upper triangular of the distance matrix
        for i in range(0, D.shape[0]):
            Total_number_of_people.add(i)
            for j in range(i + 1, D.shape[1]):
                if D[i, j] < MIN_DISTANCE:
                    people_index_violation.add(i)
                    people_index_violation.add(j)
                    # If yes violated the proximity then add those pair of people in people_index_violation set

    #Loop over the results
    for (i, (prob, bbox, centroid)) in enumerate(results):
            #Now what we will do here is basically change the color of bounding box for people in people_index_violation set and set color to red while for other color will be green
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (0, 255, 0)

        #If the index is there in people_index_violation set then change color to red of bounding box
        if i in people_index_violation:
            color = (0, 0, 255)

        #Draw the bounding box around the person and a circle around the centroid coordinate od a person
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.circle(frame, (cX, cY), 5, color, 1)


    #Displaying the text showing total number of people in frame and total number of people under violation
    text = "Peoples violationing social distancing: {}".format(len(people_index_violation))
    cv2.putText(frame, text, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
    """text = "Total number of people in frame are: {}".format(len(Total_number_of_people))
    cv2.putText(frame, text, (10, frame.shape[0] - 25),
        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)"""
    
    
    writer = cv2.VideoWriter("output_livestream.avi", cv2.VideoWriter_fourcc(*"MJPG"), 25,(frame.shape[1], frame.shape[0]), True)
    writer.write(frame)


    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
         break
cap.release()
#out.release()
cv2.destroyAllWindows()

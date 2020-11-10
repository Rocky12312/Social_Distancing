#Here we will use yolo algorithm which actually gives the centre of bounding boxes along with height and width and we will utilise these outputs for finding the euclidean distance between the pairs--
#of people in frame as well as for creating the best fit bounding box around the person.

#Initializing the threshold when applying non-maxima suppression(Actually we use non maxima suppression(NMS) to remove the bounding boxes which have probablity less than set threshold)
NMS_THRESH = 0.3
#Initializing the minimum probability for filtering the weak detections
MIN_CONF = 0.3

#Importing the necessary required Libraraies
import os
import cv2
import numpy as np


#Function for detecting peoples in frames
def detect_people(frame, net, ln, personIdx=0):
	#Dimensions of the frame
	(H, W) = frame.shape[:2]
	results = []

	#Preprocessing the frames which will requires the blob construction from there we will be able to perform object detection with Yolo and OpenCV
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
	net.setInput(blob)
	layerOutputs = net.forward(ln)

	#Initializing the lists of detected bounding boxes
	boxes = []
	#Centroid
	centroids = []
	#Confidences
	confidences = []

	#Looping over each of the layer outputs
	for output in layerOutputs:
		#Looping over each of the detections
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			if classID == personIdx and confidence > MIN_CONF:
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				#Using the center (x, y) coordinates to derive the top and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				#Updating the list of bounding box coordinates centroids and confidences
				boxes.append([x, y, int(width), int(height)])
				centroids.append((centerX, centerY))
				confidences.append(float(confidence))

	#Using NMS to suppress weak box predictions and make sure at least one prediction exist
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH)

	
	if len(idxs) > 0:
		for i in idxs.flatten():
			#Extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# update our results list to consist of the person
			# prediction probability, bounding box coordinates,
			# and the centroid
			#In this step we will append the confidence(person prediction probablity ,the bounding box coordinates and the centroid which we will use later to find euclidean distance)
			r = (confidences[i], (x, y, x + w, y + h), centroids[i])
			#Appending all of the above into results
			results.append(r)

	return results

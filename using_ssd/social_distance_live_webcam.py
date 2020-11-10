import cv2
import numpy as np
from math import pow, sqrt


caffeNetwork = cv2.dnn.readNetFromCaffe("./SSD_MobileNet_prototxt.txt", "./SSD_MobileNet.caffemodel")
#Class Labels on which it is trained are
#background
#aeroplane
#bicycle
#bird
#boat
#bottle
#bus
#car
#cat
#chair
#cow
#diningtable
#dog
#horse
#motorbike
#person
#pottedplant
#sheep
#sofa
#train
#tvmonitor


cap = cv2.VideoCapture(0)
#Check if camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video stream or file")
while(cap.isOpened()):
    ret, frame = cap.read()
    highRisk = set()
    position = dict()
    detectionCoordinates = dict()
    if not ret:
        break
    
    
    #Using CLAHE preprocessing algorithm for better person detection
    #CLAHE uses value channel of hsv
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Value channel refers to the lightness or darkness of a colour.
    # Image without hue or saturation is a grayscale image.
    hsv_planes = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    hsv_planes[2] = clahe.apply(hsv_planes[2])
    hsv = cv2.merge(hsv_planes)
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    (imageHeight, imageWidth) = frame.shape[:2]
    pDetection = cv2.dnn.blobFromImage(cv2.resize(frame, (imageWidth, imageHeight)), 0.007843, (imageWidth, imageHeight), 127.5)

    caffeNetwork.setInput(pDetection)
    #Getting the detections
    detections = caffeNetwork.forward()

    for i in range(detections.shape[2]):
        accuracy = detections[0, 0, i, 2]
        #accuracy > accuracy_threshold(take between o to 1)
        if accuracy > 0.4:
            # Detection class and detection box coordinates.
            idOfClasses = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([imageWidth, imageHeight, imageWidth, imageHeight])
            (startX, startY, endX, endY) = box.astype('int')

            #If the class detcted is person(then only we will make bounding box)
            if idOfClasses == 15.00:
                # Default drawing bounding box.
                bboxDefaultColor = (255,255,255)
                cv2.rectangle(frame, (startX, startY), (endX, endY), bboxDefaultColor, 2)
                detectionCoordinates[i] = (startX, startY, endX, endY)
                #Can also declare below variable as global
                # calculateConstant_x = 300
                # calculateConstant_y = 615
                # Centroid of bounding boxes
                centroid_x = round((startX + endX) / 2, 4)
                centroid_y = round((startY + endY) / 2, 4)
                bboxHeight = round(endY - startY, 4)
                distance = (300 * 615) / bboxHeight
                #Centroid in centimeter distance
                centroid_x_centimeters = (centroid_x * distance) / 615
                centroid_y_centimeters = (centroid_y * distance) / 615
                position[i] = (centroid_x_centimeters, centroid_y_centimeters, distance)

    #Peoples violating rules
    for i in position.keys():
        for j in position.keys():
            if i < j:
                distanceOfBboxes = sqrt(pow(position[i][0]-position[j][0],2) + pow(position[i][1]-position[j][1],2) + pow(position[i][2]-position[j][2],2))
                if distanceOfBboxes < 100:# 100cm or lower
                    highRisk.add(i)
                    highRisk.add(j)
       

    cv2.putText(frame, "Persons in violation of sdn: " + str(len(highRisk)) , (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, "Total person detcted : " + str(len(detectionCoordinates)), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for i in position.keys():
        if i in highRisk:
            rectangleColor = (0,0,225)#RED
        else:
            rectangleColor = (0,225,0)#GREEN
        (startX, startY, endX, endY) = detectionCoordinates[i]
        cv2.rectangle(frame, (startX, startY), (endX, endY), rectangleColor, 2)


    writer = cv2.VideoWriter("output_recorded_videos.avi", cv2.VideoWriter_fourcc(*"MJPG"), 25,(frame.shape[1], frame.shape[0]), True)
    writer.write(frame)
    cv2.imshow('Result', frame)
    waitkey = cv2.waitKey(1)
    if waitkey == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


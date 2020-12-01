import cv2
import numpy as np
from imutils.video import VideoStream
import time
from imutils.video import FPS

with open("classes.txt", "r") as f:
    classes = f.read().splitlines()

net = cv2.dnn.readNet('yolov3-tiny_training_4000.weights', 'yolov3-tiny_training.cfg')
# net = cv2.dnn.readNet('yolov3_6000.weights', 'yolov3_training.cfg')
cap = VideoStream(src=1).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2)
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))
fps = FPS().start()

while True:
    img = cap.read()
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append((float(confidence)))
            class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)
    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence_float= round(confidences[i],2)
            confidence = str(round(confidences[i],2))
            if confidence_float > 0.8 and label=='Wearing Mask':
                color = (0,255,0)
                cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)
                fps.update()
            if confidence_float < 0.8 or label=='Not Wearing Mask':
                color = (0, 0, 255)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, "put mask on ", (x, y + 20), font, 2, (255, 255, 255), 2)

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)

    if key==27:
        break
    fps.update()
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))






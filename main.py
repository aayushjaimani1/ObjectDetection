import numpy as np
import cv2

thres = 0.5 # Threshold to detect object
nms_threshold = 0.2 #(0.1 to 1) 1 means no suppress , 0.1 means high suppress
cap = cv2.VideoCapture('Road_traffic_video2.mp4')
cap.set(cv2.CAP_PROP_FRAME_WIDTH,280) #width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,120) #height
cap.set(cv2.CAP_PROP_BRIGHTNESS,150) #brightness

classNames = []
with open('coco.names','r') as f:
    classNames = f.read().splitlines()
print(classNames)
font = cv2.FONT_HERSHEY_PLAIN
#font = cv2.FONT_HERSHEY_COMPLEX
Colors = np.random.uniform(0, 255, size=(len(classNames), 3))

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success,img = cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1,-1)[0])
    confs = list(map(float,confs))
    # print(type(confs[0]))
    # print(confs)
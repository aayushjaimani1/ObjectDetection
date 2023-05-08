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

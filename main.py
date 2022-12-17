import cv2
import numpy as np
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from skimage.morphology import binary_closing

cap = cv2.VideoCapture("balls.mp4")
cv2.namedWindow("Camera", cv2.WINDOW_KEEPRATIO)

while cap.isOpened():
    _, frame = cap.read()
    
    if _:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        _, thresh = cv2.threshold(gray,0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.3, 20, param1=50, param2=50, minRadius=50, maxRadius=130)
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
                # draw the outer circle
                cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
                # draw the center of the circle
                cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)
                
        
        cv2.imshow('Camera', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()











    # blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    # binary = frame.copy()
    # binary = frame.mean(2)
    
    
        
    # _, thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # for i in len(contours):
    #     cv2.drawContours(frame, contours, i, (255, 0, 0), 6) 
import cv2
import numpy as np
import random
import time
import math

cv2.namedWindow("Camera", cv2.WINDOW_KEEPRATIO)
cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
cam.set(cv2.CAP_PROP_EXPOSURE, -4)


x_prv = 0 
y_prv = 0

cur_time = 0
def contours(m):
    return cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

def circle(c):
    global x_prv
    global y_prv
    if len(c) > 0:
        c = max(c, key=cv2.contourArea)
        (x,y), radius = cv2.minEnclosingCircle(c)
        current_time = round(time.time()*1000)
        k = radius*2/73
        distance = (math.sqrt((x - x_prv)**2+(y-y_prv)**2))
        speed = k*distance/current_time
        x_prv = x
        y_prv = y
        cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 255), 2)

        cv2.putText(frame, f"Speed of {speed} meters in sec", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (10, 10, 10))

        colour_arr = hsv[int(y),int(x)]
        if lower_blue[0] <= colour_arr[0] <= upper_blue[0] \
         and lower_blue[1] <= colour_arr[1] <= upper_blue[1] \
         and lower_blue[1] <= colour_arr[1] <= upper_blue[1]:
         if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)

        
def result(m):
    return cv2.bitwise_and(frame, frame, mask = m)

measures = []
hsv = []
while cam.isOpened():
    ret, frame = cam.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([60,180,40])
    upper_blue = np.array([120,260,113])

    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    contours_blue, _ = contours(mask_blue)

    res_b = result(mask_blue)

    circle(contours_blue)

    cv2.imshow("Camera", frame)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
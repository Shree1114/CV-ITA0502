import cv2
import numpy as np
path = ""
img = cv2.imread(path)
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(img,(7,7),10) #10 is blur value
imgCanny = cv2.Canny(imgBlur,100,200)
cv2.imshow("GrayScale",imgCanny)
cv2.waitKey(0)

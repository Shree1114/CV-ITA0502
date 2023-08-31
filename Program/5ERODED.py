import cv2
import numpy as np
kernel = np.ones((5,5),np.uint8)
path = ""
img = cv2.imread(path)
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(img,(7,7),10) #10 is blur value
imgCanny = cv2.Canny(imgBlur,100,200)
imgDilation = cv2.dilate(imgCanny,kernel,iterations = 10)
imgEroded = cv2.erode(imgDilation,kernel,iterations=2)
#cv2.imshow("GrayScale",imgDilation)
cv2.imshow("GrayScale",imgEroded)

cv2.waitKey(0)

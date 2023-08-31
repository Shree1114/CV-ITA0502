import cv2
import numpy as np
# Read the image
image_path = ""
img = cv2.imread(image_path)
# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detect faces in the image
faces = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml').detectMultiScale(gray, 1.3, 5)
eyes = cv2.CascadeClassifier('haarcascade_eye.xml').detectMultiScale(gray,1.2,5)
# Draw a rectangle around each face
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

"""for (x,y,w,h) in eyes:
    cv2.rectangle(img, (x,y), (x + w, y + h), (0, 255, 0), 2)"""
# Show the image
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np

image= cv2.imread('test_image.jpg')

cv2.imshow('result', image)#this function should be followed by the waitkey()
#cv2.waitKey(0)#this function display the image for the specified time in ms

lane_image= np.copy(image) #because arrays are immutable meaning if assigned any change in one of them will affect the other
gray= cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
cv2.imshow('gray', gray)
#cv2.waitKey(0)

#This is an optional because canny function also do it
blur= cv2.GaussianBlur(gray, (5,5),0)#kernel size is 5*5 which is suitable for most cases
cv2.imshow('blur', blur)
cv2.waitKey(0)

 #Canny function will perform a drevative on both x, and y directions
 #to indicate the change intensity
 #cv2.Canny(image, low_threshold, high_threshold)
 #black areas -> low changes in intensity
 #white lines -> high change in intesity exceeding te threshold


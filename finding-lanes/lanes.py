import cv2

image= cv2.imread('test_image.jpg')
cv2.imshow('result', image)#this function should be followed by the waitkey()
cv2.waitkey(0)#this function display the image for the specified time in ms

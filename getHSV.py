# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 22:33:48 2014

@author: Argen
"""

'''
Sliders to determine best ranges for image
'''
import cv2
import numpy as np

#cap = cv2.VideoCapture(0)
img = 'images/images_rojo/c1_image00.png'
#img = 'images/images_rojo/c2_image00.png'
#img = 'images/images_verde/c2_image00.png'
im = cv2.imread(img)
im1 = cv2.resize(im,(640,480))
im1 = cv2.GaussianBlur(im1,(5,5),0)

#Original image
cv2.imshow("Original Image",im1)

# Convert BGR to HSV
hsv = cv2.cvtColor(im1, cv2.COLOR_BGR2HSV)
#hsv = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)

# define range of blue color in HSV
#blue
lower = np.array([ 92,  39, 146],dtype="uint8")
upper = np.array([114,  95, 255],dtype="uint8")

#red
#lower = np.array([0,6,228],dtype="uint8")
#upper = np.array([179,122,255],dtype="uint8")

#green
#lower = np.array([ 37, 16, 186],dtype="uint8")
#upper = np.array([118, 184, 255],dtype="uint8")


# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv,lower, upper)

# Bitwise-AND mask and original image
res = cv2.bitwise_and(im1,im1, mask= mask)


def nothing(x):
    pass

cv2.imshow("HSV result",res)
cv2.createTrackbar("Hl","HSV result",lower[0],179,nothing)
cv2.createTrackbar("Sl","HSV result",lower[1],255,nothing)
cv2.createTrackbar("Vl","HSV result",lower[2],255,nothing)
cv2.createTrackbar("Hu","HSV result",upper[0],179,nothing)
cv2.createTrackbar("Su","HSV result",upper[1],255,nothing)
cv2.createTrackbar("Vu","HSV result",upper[2],255,nothing)
    
while(1):
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    mask = cv2.inRange(hsv,lower, upper)
    res = cv2.bitwise_and(im1,im1, mask= mask)    
    cv2.imshow("HSV result",res)
    
    lower[0]=cv2.getTrackbarPos("Hl","HSV result")
    lower[1]=cv2.getTrackbarPos("Sl","HSV result")
    lower[2]=cv2.getTrackbarPos("Vl","HSV result")
    upper[0]=cv2.getTrackbarPos("Hu","HSV result")
    upper[1]=cv2.getTrackbarPos("Su","HSV result")
    upper[2]=cv2.getTrackbarPos("Vu","HSV result")
    
print 'np.array(['+np.str(lower[0])+', '+np.str(lower[1])+', '+np.str(lower[2])+'])'
print upper    
cv2.destroyAllWindows()
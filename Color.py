# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 12:20:11 2014

@author: Argen
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = 'images_azul/c2_image0.png'
im1 = cv2.imread(img1)
#im1 = cv2.resize(im1,(640,480))
im1 = cv2.GaussianBlur(im1,(5,5),0)
im1 = im1[700:900,600:1600]
cv2.cv.NamedWindow("Masked image",cv2.WINDOW_NORMAL)
#cv2.imshow("Masked image",im1)
#cv2.cv.ResizeWindow("Masked image",640,480)

#Histogram calculation
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([im1],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.title("Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.show()

plt.figure()
im2 = im1[:,:,::-1]
plt.imshow(im2)


# Convert BGR to HSV
hsv = cv2.cvtColor(im1, cv2.COLOR_BGR2HSV)

# define range of blue color in HSV
lower_blue = np.array([ 92,  39, 146],dtype="uint8")
upper_blue = np.array([114,  95, 255],dtype="uint8")

lower_red = np.array([ 0, 6, 228],dtype="uint8")
upper_red = np.array([179, 122, 255],dtype="uint8")

lower_green = np.array([ 37, 16, 186],dtype="uint8")
upper_green = np.array([118, 184, 255],dtype="uint8")


# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv,lower_blue, upper_blue)
mask1 = cv2.inRange(im1,lower_blue,upper_blue)

# Bitwise-AND mask and original image
res = cv2.bitwise_and(im1,im1, mask= mask)
res2 = cv2.bitwise_and(im1,im1,mask=mask1)

cv2.cv.NamedWindow("HSV result",cv2.WINDOW_NORMAL)
cv2.imshow("HSV result",np.vstack((res,im1)))
cv2.waitKey(0)    
cv2.destroyAllWindows()


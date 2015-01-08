# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 18:50:11 2014

@author: Argen
"""
import cv2
import numpy as np

img1 = 'images_azul/c2_image00.png'
im1 = cv2.imread(img1)
#im1 = cv2.resize(im1,(640,480))
im1 = cv2.GaussianBlur(im1,(5,5),0)
im1 = im1[750:800,1100:1200]

im3 = im1.copy()
gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

#detect circles
circles = cv2.HoughCircles(gray, cv2.cv.CV_HOUGH_GRADIENT, 1, 5)
# ensure at least some circles were found
if circles is not None:
    print "Found circle"
    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")
    
    # loop over the (x, y) coordinates and radius of the circles
    for (x, y, r) in circles:
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        cv2.circle(im3, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(im3, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

        # show the output image
        cv2.imshow("output", np.hstack([im1, im3]))
        cv2.waitKey(0)
cv2.destroyAllWindows()
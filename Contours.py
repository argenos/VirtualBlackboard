# -*- coding: utf-8 -*-
"""
Created on Sat Jan 10 22:32:41 2015

@author: Argen
"""

import numpy as np
import cv
import cv2
import ROISelection as r
import Color as color

im = cv2.imread('images/images_azul/c1_image40.png')
im = cv2.GaussianBlur(im, (3, 3), 0)
#im = cv2.resize(im,(640,480))

x1, y1, x2, y2 = r.getROI(im)

im = im[y1:y2,x1:x2]
orig = im.copy()
color.color_mask(im)
im = color.getColorMask(im,'b')

cv2.imshow("Color",im)
cv2.waitKey(0)
cv2.destroyAllWindows()

imgray = cv2.cvtColor(im,cv2.COLOR_HSV2BGR)
imgray = cv2.cvtColor(imgray,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,cv2.THRESH_BINARY)

cv2.imshow("Threshold",thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''
circles = cv2.HoughCircles(imgray,cv.CV_HOUGH_GRADIENT,1,imgray.shape[0]/4,
                            param1=140,param2=15,minRadius=1,maxRadius=40)
                            
if circles != None:
    circles = np.uint16(np.around(circles))
    print circles.shape()
else:
    print "No circles detected"
    

for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(imgray,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(imgray,(i[0],i[1]),2,(0,0,255),3)
cv2.imshow("Circles",imgray)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''


#contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

#cv2.drawContours(imgray, contours, -1, (255,0,0), 4)


i = 0
bigArea = 0
for cnt in contours:
   area = cv2.contourArea(cnt)
   if area>= bigArea:
       bigArea=i
   i = i + 1
   

cnt = contours[i-1]
#cv2.drawContours(imgray, [cnt], 0, (255,255,255), 3)

(x,y),radius = cv2.minEnclosingCircle(cnt)
center = (int(x),int(y))
radius = int(radius)
cv2.circle(orig, center, radius, (10,10,255), 3)

cv2.namedWindow("Contours", cv2.WINDOW_NORMAL)
cv2.imshow("Contours", orig)

cv2.waitKey(0)
cv2.destroyAllWindows()

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 17:17:07 2014

@author: Argen
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = 'images_azul/c2_image0.png'
im1 = cv2.imread(img1)
#im1 = cv2.resize(im1,(640,480))
im1 = cv2.GaussianBlur(im1,(5,5),0)
im1 = im1[750:800,1100:1200]
#cv2.cv.NamedWindow("Masked image",cv2.WINDOW_NORMAL)

plt.figure()
im2 = im1[:,:,::-1]
plt.imshow(im2)


#Histogram calculation
plt.figure()
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([im1],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.title("Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")
plt.show()
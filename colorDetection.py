# -*- coding: utf-8 -*-
"""
Created on Sun Jan 11 19:26:46 2015

@author: Argen
"""
import cv2
import Contour42 as contour
import numpy as np
from matplotlib import pyplot as plt


def getMarker(img,roi,c, show=False):
    org,im,crop,center,radius = contour.getImgWithMarker(img,roi,c,True)
    y1 = center[1]-radius
    y2 = center[1]+radius
    x1 = center[0]-radius
    x2 = center[0]+radius

    marker = crop[y1:y2,x1:x2]
    if show:
        cv2.imshow('Marker',marker)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return marker, center

def getHistogram(img):
    plt.figure()
    im2 = img[:,:,::-1]
    plt.imshow(im2)

    #Histogram calculation
    plt.figure()
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.title("Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.show()

def getColor(marker):
    color = np.zeros(marker.shape,np.uint8)
    color[:,:,0]=marker[marker.shape[0]/2,marker.shape[1]/2,0]
    color[:,:,1]=marker[marker.shape[0]/2,marker.shape[1]/2,1]
    color[:,:,2]=marker[marker.shape[0]/2,marker.shape[1]/2,2]

    return color

def main():
    ex = cv2.imread('images/images_azul/c2_image00.png')

    ex = cv2.GaussianBlur(ex, (3, 3), 0)
    roi,roi2 = contour.init(ex)
    marker, center = getMarker(ex,roi,'b',True)
    getHistogram(marker)

    color = getColor(marker)


    cv2.imshow("Color",color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

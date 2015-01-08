# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 12:20:11 2014

@author: Argen
"""

import cv2
import numpy as np
import scipy.optimize as opt
import scipy as scp
from matplotlib import pyplot as plt


# define range of blue color in HSV
#lower_blue = np.array([92,  39, 146], dtype="uint8")
#upper_blue = np.array([114,  95, 255], dtype="uint8")
lower_blue = np.array([92,  20, 110], dtype="uint8")
upper_blue = np.array([114,  100, 255], dtype="uint8")
# define range of red color in HSV
lower_red = np.array([0, 6, 228], dtype="uint8")
upper_red = np.array([179, 122, 255], dtype="uint8")
# define range of green color in HSV
lower_green = np.array([37, 16, 186], dtype="uint8")
upper_green = np.array([118, 184, 255], dtype="uint8")


def color_mask(image):
    im1 = cv2.GaussianBlur(image, (5, 5), 0)
    #histogram_rgb(im1)
    # Convert BGR to HSV
    hsv = cv2.cvtColor(im1, cv2.COLOR_BGR2HSV)
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(im1, im1, mask=mask)

    cv2.namedWindow("HSV MASK", cv2.WINDOW_NORMAL)
    cv2.imshow("HSV MASK", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def histogram_rgb(image):
    #Histogram calculation
    plt.figure()
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([0, 256])
    plt.title("Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.show()


def histogram_hsv(image1, image2):
    im = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([im], [0], None, [180], [0, 180])
    s = cv2.calcHist([im], [1], None, [256], [0, 256])
    v = cv2.calcHist([im], [2], None, [256], [0, 256])
    im = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
    h += cv2.calcHist([im], [0], None, [180], [0, 180])
    s += cv2.calcHist([im], [1], None, [256], [0, 256])
    v += cv2.calcHist([im], [2], None, [256], [0, 256])

    plt.figure()
    plt.plot(h, 'r', label='Hue')
    plt.plot(s, 'g', label='Saturation')
    plt.plot(v, 'b', label='Value')
    plt.title("Histogram HSV")
    plt.legend()
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.show()

    fit_gaussian(h)
    fit_gaussian(s)
    fit_gaussian(v)


def fit_gaussian(histogram):
    n = len(histogram)
    y = histogram/np.max(histogram)
    x = np.reshape(np.linspace(0, n-1, num=n), (n,1))
    mean = (np.sum(np.multiply(x, y))/np.sum(y))
    sigma = np.sqrt(np.sum(y*(x-mean)**2)/n)
    print "Mean: ", mean
    print "Sigma: ", sigma

    print "Median: ", np.median(histogram)
    print "Mean: ", np.mean(histogram)
    print "Max: ", np.max(histogram)
    print [i for i, x in enumerate(histogram) if x == np.max(histogram)]

    return mean, sigma


def histogram_ycc(image):
    im = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    cv2.namedWindow("YCrCb", cv2.WINDOW_NORMAL)
    cv2.imshow("YCrCb", im)
    cv2.namedWindow("Luminence", cv2.WINDOW_NORMAL)
    cv2.imshow("Luminence", im[:, :, 0])
    cv2.namedWindow("CrominanceRed", cv2.WINDOW_NORMAL)
    cv2.imshow("CrominanceRed", im[:, :, 1])
    cv2.namedWindow("CrominanceBlue", cv2.WINDOW_NORMAL)
    cv2.imshow("CrominanceBlue", im[:, :, 02])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    print "Color Masking"


if __name__ == "__main__":
    main()


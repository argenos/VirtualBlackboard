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
lower_blue = np.array([92,  39, 146], dtype="uint8")
upper_blue = np.array([114,  95, 255], dtype="uint8")
# define range of red color in HSV
lower_red = np.array([0, 0, 230], dtype="uint8")
upper_red = np.array([179, 255, 255], dtype="uint8")
# define range of green color in HSV
lower_green = np.array([37, 16, 186], dtype="uint8")
upper_green = np.array([118, 184, 255], dtype="uint8")


def color_mask(image, color, display):
    #im1 = cv2.GaussianBlur(image, (5, 5), 0)
    im1 = image.copy()
    # Convert BGR to HSV
    hsv = cv2.cvtColor(im1, cv2.COLOR_BGR2HSV)
    # Threshold the HSV image to get only blue colors
    if color == 'b':
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
    elif color == 'r':
        mask = cv2.inRange(hsv, lower_red, upper_red)
    elif color == 'g':
        mask = cv2.inRange(hsv, lower_green, upper_green)
    # Bitwise-AND mask and original image
    color_masked = cv2.bitwise_and(im1, im1, mask=mask)

    if display:
        cv2.namedWindow("HSV MASK", cv2.WINDOW_NORMAL)
        cv2.imshow("HSV MASK", color_masked)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return color_masked


def getColorMask(image, c):
    im1 = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(im1, cv2.COLOR_BGR2HSV)
    if c == 'b':
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
    elif c == 'r':
        mask = cv2.inRange(hsv, lower_red, upper_red)
    elif c == 'g':
        mask = cv2.inRange(hsv, lower_green, upper_green)

    res = cv2.bitwise_and(im1, im1, mask=mask)
    
    return res
    

def back_projection(frame):
    src1_histogram = cv2.imread("images/blue_sphere_00.png")
    src2_histogram = cv2.imread("images/blue_sphere_01.png")
    src1_histogram = cv2.cvtColor(src1_histogram, cv2.COLOR_BGR2HSV)
    src2_histogram = cv2.cvtColor(src2_histogram, cv2.COLOR_BGR2HSV)

    h1 = cv2.calcHist([src1_histogram], [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(h1, h1, 0, 255, cv2.NORM_MINMAX)
    h2 = cv2.calcHist([src2_histogram], [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(h2, h2, 0, 255, cv2.NORM_MINMAX)

    histogram = np.add(h1, h2)

    hue = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    backproj = cv2.calcBackProject([hue], [0, 1], h1, [0, 180, 0, 256], 1)

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cv2.filter2D(backproj, -1, disc, backproj)

    # threshold and binary AND
    ret, thresh = cv2.threshold(backproj, 100, 255, 0)
    thresh = cv2.merge((thresh, thresh, thresh))
    res = cv2.bitwise_and(frame, thresh)

    cv2.namedWindow("BackProjection", cv2.WINDOW_NORMAL)
    cv2.imshow("BackProjection", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return res


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
    s = cv2.calcHist([im], [1], None, [255], [0, 256])
    v = cv2.calcHist([im], [2], None, [255], [0, 256])
    im = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
    h += cv2.calcHist([im], [0], None, [180], [0, 180])
    s += cv2.calcHist([im], [1], None, [255], [0, 256])
    v += cv2.calcHist([im], [2], None, [255], [0, 256])

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


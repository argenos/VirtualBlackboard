# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 12:20:11 2014

@author: Argen
"""
import cv2
import numpy as np
import circleDetection as Detector
from matplotlib import pyplot as plt
import HSV as h
import ROI42 as roi

# define range of blue color in HSV
lower_blue = np.array([92,  39, 146], dtype="uint8")
upper_blue = np.array([114,  95, 255], dtype="uint8")
# New values
# lower_blue = np.array([60, 41, 20], dtype="uint8")
# upper_blue = np.array([121, 202, 210], dtype="uint8")

# define range of red color in HSV
lower_red = np.array([0, 0, 230], dtype="uint8")
upper_red = np.array([179, 255, 255], dtype="uint8")
# New values
# lower_red = np.array([0, 67, 61], dtype="uint8")
# upper_red = np.array([179, 144, 255], dtype="uint8")

# define range of green color in HSV
lower_green = np.array([37, 16, 186], dtype="uint8")
upper_green = np.array([118, 184, 255], dtype="uint8")
# New Values
# lower_green = np.array([38, 70, 86], dtype="uint8")
# upper_green = np.array([130, 145, 226], dtype="uint8")

# Yellow
# lower_yellow = np.array([29, 35, 67], dtype="uint8")
# upper_yellow = np.array([57, 58, 255], dtype="uint8")

# White
# lower_white = np.array([0, 0, 224], dtype="uint8")
# upper_white = np.array([48, 18, 255], dtype="uint8")

# Global boundaries
global_lower_side = []
global_upper_side = []
global_lower_top = []
global_upper_top = []
global_lower = []
global_upper = []

def track_hsv(image):
    # im1 = cv2.resize(image, (640, 480))
    im1 = image.copy()
    im1 = cv2.GaussianBlur(im1, (5, 5), 0)
    # Convert BGR to HSV
    hsv = cv2.cvtColor(im1, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower = np.array([0, 0, 0], dtype="uint8")
    upper = np.array([179, 255, 255], dtype="uint8")

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower, upper)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(im1, im1, mask=mask)

    def nothing(x):
        pass

    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    cv2.imshow("Result", res)
    cv2.namedWindow("Color Thresholds")
    cv2.createTrackbar("Hl", "Color Thresholds", lower[0], 179, nothing)
    cv2.createTrackbar("Sl", "Color Thresholds", lower[1], 255, nothing)
    cv2.createTrackbar("Vl", "Color Thresholds", lower[2], 255, nothing)
    cv2.createTrackbar("Hu", "Color Thresholds", upper[0], 179, nothing)
    cv2.createTrackbar("Su", "Color Thresholds", upper[1], 255, nothing)
    cv2.createTrackbar("Vu", "Color Thresholds", upper[2], 255, nothing)

    while True:
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        mask = cv2.inRange(hsv, lower, upper)
        res = cv2.bitwise_and(im1, im1, mask=mask)
        cv2.imshow("Result", res)

        lower[0] = cv2.getTrackbarPos("Hl", "Color Thresholds")
        lower[1] = cv2.getTrackbarPos("Sl", "Color Thresholds")
        lower[2] = cv2.getTrackbarPos("Vl", "Color Thresholds")
        upper[0] = cv2.getTrackbarPos("Hu", "Color Thresholds")
        upper[1] = cv2.getTrackbarPos("Su", "Color Thresholds")
        upper[2] = cv2.getTrackbarPos("Vu", "Color Thresholds")

    print "Lower: ", lower
    print "Upper: ", upper

    print 'Lower: np.array([%d, %d, %d], dtype="uint8")'%(lower[0],lower[1],lower[2])
    print 'Upper: np.array([%d, %d, %d], dtype="uint8")'%(upper[0],upper[1],upper[2])

    cv2.destroyAllWindows()




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
    else:
        print "No color recognized. Use 'R', 'G' or 'B' when calling the function.\n"
        return -1
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

def getAutoColorMask(image,boundaries='combined'):
    im1 = cv2.GaussianBlur(image, (5, 5), 0)
    hsv = cv2.cvtColor(im1, cv2.COLOR_BGR2HSV)
    if boundaries == 'combined':
        mask = cv2.inRange(hsv, global_lower, global_upper)
    elif boundaries == 'side':
        mask = cv2.inRange(hsv, global_lower_side, global_upper_side)
    elif boundaries == 'top':
        mask = cv2.inRange(hsv, global_lower_top, global_upper_top)

    res = cv2.bitwise_and(im1,im1,mask=mask)
    return res


def initializeBoundaries(imgside, imgtop):
    global global_lower, global_lower_side, global_lower_top, global_upper, global_upper_side, global_upper_top
    im1 = cv2.GaussianBlur(imgtop,(5,5),0)
    l, u = h.getHSV(im1)
    global_lower_top = l
    global_upper_top = u
    res1 = h.applyHSV(imgtop,l,u)
    cv2.imshow('Result HSV',res1)

    im2 = cv2.GaussianBlur(imgside,(5,5),0)
    l2,u2 = h.getHSV(im2)
    global_lower_side = l2
    global_upper_side = u2
    res2 = h.applyHSV(imgside,l2,u2)

    global_lower = np.min(np.vstack((global_lower_side,global_lower_top)),axis=0)
    global_upper = np.max(np.vstack((global_upper_side,global_upper_top)),axis=0)
    print global_lower,global_upper

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return res1,res2


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


def compute_histogram(original_image, color_space):

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

    def histogram_hsv(image):
        im = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([im], [0], None, [180], [0, 180])
        s = cv2.calcHist([im], [1], None, [255], [0, 256])
        v = cv2.calcHist([im], [2], None, [255], [0, 256])

        plt.figure()
        plt.plot(h, 'r', label='Hue')
        plt.plot(s, 'g', label='Saturation')
        plt.plot(v, 'b', label='Value')
        plt.title("Histogram HSV")
        plt.legend()
        plt.xlabel("Bins")
        plt.ylabel("# of Pixels")
        plt.show()

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

    if color_space == 'rgb':
        histogram_rgb(original_image)
    elif color_space == 'hsv':
        histogram_hsv(original_image)
    elif color_space == 'ycrcb':
        histogram_ycc(original_image)
    else:
        return -1


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


def equalize_component(image, component):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Image", np.hstack((image, hsv)))
    cv2.waitKey(0)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    cv2.namedWindow("HSV Components", cv2.WINDOW_NORMAL)
    out = np.hstack((h, np.hstack((s, v))))
    cv2.imshow("HSV Components", out)
    cv2.waitKey(0)

    hsv2 = hsv.copy()
    if component == 'h':
        n = cv2.equalizeHist(hsv2[:, :, 0])
        new_image = np.dstack((n, np.dstack((s, v))))
    elif component == 's':
        n = cv2.equalizeHist(hsv2[:, :, 1])
        new_image = np.dstack((h, np.dstack((n, v))))
    elif component == 'v':
        n = cv2.equalizeHist(hsv2[:, :, 2])
        new_image = np.dstack((h, np.dstack((s, n))))
    else:
        new_image = np.dstack((h, np.dstack((s, v))))

    new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2BGR)

    cv2.namedWindow("HSV MOD", cv2.WINDOW_NORMAL)
    cv2.imshow("HSV MOD", np.hstack((image, new_image)))
    cv2.waitKey(0)

    return new_image


def main():
    print "Color Masking"
    directory = "images/images_azul/"
    frame = cv2.imread(directory+"c1_image00.png")
    # test = cv2.imread("images/blue_sphere_00.png")
    # track_hsv(frame)
    # color_mask(frame, color='b', display=True)
    # compute_histogram(test, 'rgb')
    # compute_histogram(test, 'hsv')
    processed = equalize_component(frame, 's')
    # track_hsv(processed)
    Detector.detect_circles(processed)

if __name__ == "__main__":
    # main()
    img1 = cv2.imread('images/frames/blue/c1_image014.png')
    img1 = cv2.resize(img1, (1200, 700))
    x1, y1, x2, y2 = roi.getROI(img1)

    im1 = img1[y1:y2, x1:x2]
    # track_hsv(img)
    img2 = cv2.imread('images/frames/blue/c2_image014.png')
    img2 = cv2.resize(img2,(1200, 700))
    x1, y1, x2, y2 = roi.getROI(img2)

    im2 = img2[y1:y2, x1:x2]

    initializeBoundaries(im1, im2)

    cv2.destroyAllWindows()
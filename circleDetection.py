__author__ = 'laverden'

import numpy as np
import cv2 as cv
import glob


def draw_one_circle(to_image, param):
    cv.circle(to_image, (param[0], param[1]), 15, (0, 0, 255), 2)
    # draw the center of the circle
    cv.circle(to_image, (param[0], param[1]), 2, (200, 20, 200), 3)

# Printing function for the found circles.
def print_circles(to_image, circles_param):
    for k in circles_param[0, :]:
        # draw the outer circle
        cv.circle(to_image, (k[0], k[1]), k[2], (0, 0, 255), 2)
        # draw the center of the circle
        cv.circle(to_image, (k[0], k[1]), 2, (200, 20, 200), 3)


def hough_circles(image):
    circles = cv.HoughCircles(image, cv.cv.CV_HOUGH_GRADIENT, dp=1, minDist=image.shape[0]/4,
                               param1=140, param2=10, minRadius=5, maxRadius=15)
    circles = np.uint16(np.around(circles))
    print circles

    return circles


def compare_circle_detection(image):
    #-----------------------------------------------------------------------------------------
    # Method 1: Hough Circle Transformation.
    #-----------------------------------------------------------------------------------------
    blur = cv.GaussianBlur(image, (5, 5), sigmaX=1, sigmaY=1)
    gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)

    circles = cv.HoughCircles(gray, cv.cv.CV_HOUGH_GRADIENT, dp=1, minDist=gray.shape[0]/4,
                              param1=140, param2=15, minRadius=2, maxRadius=15)

    circles = np.uint16(np.around(circles))
    #print circles
    c_image = image.copy()
    print_circles(c_image, circles)

    #-----------------------------------------------------------------------------------------
    # Method 2: Hough Circles Transformation + Morphological Operations
    #-----------------------------------------------------------------------------------------
    filtered = cv.GaussianBlur(image, (3, 3), sigmaX=0, sigmaY=0)
    gray_im = cv.cvtColor(filtered, cv.COLOR_BGR2GRAY)
    er = cv.erode(gray_im, kernel=(5, 5), iterations=10)
    dil = cv.dilate(er, kernel=(5, 5), iterations=10)
    canny = cv.Canny(dil, threshold1=30, threshold2=130, L2gradient=True)
    final = cv.GaussianBlur(canny, (3, 3), sigmaX=0, sigmaY=0)

    circles2 = cv.HoughCircles(dil, cv.cv.CV_HOUGH_GRADIENT, dp=1, minDist=final.shape[0]/4,
                               param1=130, param2=5, minRadius=5, maxRadius=15)

    circles2 = np.uint16(np.around(circles2))
    #print circles
    c2_image = image.copy()
    print_circles(c2_image, circles2)

    #-----------------------------------------------------------------------------------------
    # Method 3: Finding Contours
    #-----------------------------------------------------------------------------------------
    c3_image = image.copy()
    contour, hierarchy = cv.findContours(final, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
    for c in xrange(len(contour)):
        cv.drawContours(c3_image, contours=contour, contourIdx=c, color=(0,10*c,0), lineType=cv.CV_AA)


    cv.namedWindow("Original-Filtered", cv.WINDOW_NORMAL)
    a = np.hstack((image, filtered))
    cv.imshow("Original-Filtered", a)

    cv.namedWindow("Erode-Dilate\nCanny-Final", cv.WINDOW_NORMAL)
    b = np.hstack((er, dil))
    c = np.hstack((canny, final))
    d = np.vstack((b, c))
    cv.imshow("Erode-Dilate\nCanny-Final", d)

    cv.namedWindow("Circles1", cv.WINDOW_NORMAL)
    cv.imshow("Circles1", c_image)
    cv.namedWindow("Circles2", cv.WINDOW_NORMAL)
    cv.imshow("Circles2", c2_image)
    cv.namedWindow("Circles3", cv.WINDOW_NORMAL)
    cv.imshow("Circles3", c3_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return circles


def detect_by_hough(directory):
    files1 = sorted(np.array(glob.glob(directory + "c1_image*.png")))
    files2 = sorted(np.array(glob.glob(directory + "c2_image*.png")))
    cv.namedWindow("Circles", cv.WINDOW_NORMAL)
    n_frames = len(files1)
    index = 0
    while True:
        src1 = cv.imread(files1[index])
        src2 = cv.imread(files2[index])
        if src1 is None or src2 is None:
            print "Couldn't read the images\n"
            return 0

        gray = cv.cvtColor(src2, cv.COLOR_BGR2GRAY)
        gray = cv.equalizeHist(gray)
        blur = cv.GaussianBlur(gray, (5, 5), sigmaX=0, sigmaY=0)
        circles = cv.HoughCircles(blur, cv.cv.CV_HOUGH_GRADIENT, dp=1, minDist=gray.shape[0]/2,
                                  param1=130, param2=15, minRadius=5, maxRadius=20)

        c_image = src2.copy()
        if circles is not None:
            circles = np.uint16(np.around(circles))
            print_circles(c_image, circles)

        cv.imshow("Circles", c_image)

        if index == n_frames-1:
            index = 0
        else:
            index += 1

        k = cv.waitKey(100) & 0XFF
        if k == 27:
            cv.destroyAllWindows()
            break


def main():
    print "Circle Detection"
    detect_by_hough("images/images_azul/")

if __name__ == "__main__":
    main()
    cv.destroyAllWindows()
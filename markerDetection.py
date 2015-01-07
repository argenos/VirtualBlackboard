__author__ = 'laverden'


import numpy as np
import cv as cv1
import cv2 as cv
import glob


def detect_circles(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (3, 3), sigmaX=1, sigmaY=1)
    circles = cv.HoughCircles(gray, cv1.CV_HOUGH_GRADIENT, dp=1, minDist=gray.shape[0]/4,
                              param1=140, param2=15, minRadius=2, maxRadius=15)

    circles = np.uint16(np.around(circles))
    #print circles
    cimage = image.copy()

    for i in circles[0,:]:
        # draw the outer circle
        cv.circle(cimage, (i[0],i[1]), i[2], (0,0,255), 2)
        # draw the center of the circle
        cv.circle(cimage, (i[0],i[1]), 2, (200,20,200), 3)

    cv.namedWindow("Circles", cv.WINDOW_NORMAL)
    cv.imshow("Circles", cimage)
    cv.waitKey(0)
    cv.destroyWindow("Circles")

    return circles


def detect_all_circles(directory):
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
        gray = cv.GaussianBlur(gray, (3, 3), sigmaX=0, sigmaY=0)
        #gray = cv.medianBlur(gray, 7)
        circles = cv.HoughCircles(gray, cv1.CV_HOUGH_GRADIENT, dp=1, minDist=gray.shape[0]/4,
                              param1=130, param2=15, minRadius=5, maxRadius=15)

        cimage = src2.copy()
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
                # draw the outer circle
                cv.circle(cimage, (i[0],i[1]), i[2], (0,0,255), 2)
                # draw the center of the circle
                cv.circle(cimage, (i[0],i[1]), 2, (200,20,200), 3)

        cv.imshow("Circles", cimage)

        if index == n_frames-1:
            index = 0
        else:
            index += 1

        k = cv.waitKey(10) & 0XFF
        if k == 27:
            cv.destroyAllWindows()
            break


def filtering(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    def nothing(x):
        pass

    def blur_filter(original_image, size):
        return cv.blur(original_image, (2*size+1, 2*size+1))

    def gaussian_filter(original_image, size):
        return cv.GaussianBlur(original_image, (2*size+1, 2*size+1), sigmaX=0, sigmaY=0)

    def median_filter(original_image, size):
        return cv.medianBlur(original_image, 2*size+1)

    def bilateral_filter(original_image, size):
        return cv.bilateralFilter(original_image, size+1, (size+1)**2, (size+1)/2)

    cv.namedWindow("Smoothing", cv.WINDOW_NORMAL)
    cv.createTrackbar("Kernel", "Smoothing", 0, 9, nothing)
    filter_type = "0: Blur\n1: Gaussian\n2: Median\n3: Bilateral"
    cv.createTrackbar(filter_type, "Smoothing", 0, 3, nothing)

    while True:
        f_type = cv.getTrackbarPos(filter_type, "Smoothing")
        kernel = cv.getTrackbarPos("Kernel", "Smoothing")
        if f_type == 0:
            im_filtered = blur_filter(image, kernel)
        elif f_type == 1:
            im_filtered = gaussian_filter(image, kernel)
        elif f_type == 2:
            im_filtered = median_filter(image, kernel)
        else:
            im_filtered = bilateral_filter(image, kernel)

        cv.imshow("Smoothing", im_filtered)

        k = cv.waitKey(1) & 0XFF
        if k == 27:
            cv.destroyAllWindows()
            break


def main():
    print "OpenCV version: ", cv.__version__
    cv.destroyAllWindows()
    dir_name = "images/images_azul/"
    files1 = np.array(glob.glob(dir_name + "c1_image*.png"))
    files2 = np.array(glob.glob(dir_name + "c2_image*.png"))
    src2 = cv.imread(files2[0])

    detect_all_circles(dir_name)
    filtering(src2)


if __name__ == "__main__":
    main()

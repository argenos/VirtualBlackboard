__author__ = 'laverden'


import numpy as np
import cv2 as cv


def filtering(image):
    color_image = image
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


def histogram_eq(image):
    im = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    eq = cv.equalizeHist(im)
    cv.namedWindow("HistogramEqualized", cv.WINDOW_NORMAL)
    cv.namedWindow("Normal Image", cv.WINDOW_NORMAL)
    cv.imshow("HistogramEqualized", eq)
    cv.imshow("Normal Image", im)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return eq


def sharpen(image, show):
    im = cv.GaussianBlur(image, (5, 5), 5)
    sharp = cv.addWeighted(image, 1.5, im, -0.5, 0)

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray2 = cv.convertScaleAbs(gray, cv.CV_32F)
    fil = cv.GaussianBlur(gray2, (5, 5), 5)
    l = cv.Laplacian(fil, cv.CV_32F)

    n = 1.5*gray2 - 0.5*fil - 0.01*5*np.multiply(gray2, 2*l)
    n = cv.convertScaleAbs(n, cv.CV_8U)

    if show:
        cv.namedWindow("Normal/Sharp", cv.WINDOW_NORMAL)
        cv.imshow("Normal/Sharp", np.hstack((image, sharp)))
        cv.namedWindow("Sharper", cv.WINDOW_NORMAL)
        cv.imshow("Sharper", n)
        cv.waitKey(0)
        cv.destroyAllWindows()

    return n


def main():
    print "Pre-processing"
    directory = "images/images_azul/"
    frame = cv.imread(directory+"c1_image00.png")

    #sharpen(frame, show=True)
    histogram_eq(frame)
    #filtering(frame)


if __name__ == "__main__":
    main()
    cv.destroyAllWindows()
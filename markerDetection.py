__author__ = 'laverden'


import numpy as np
import cv as cv1
import cv2 as cv
import glob
import Color as color
import ROISelection as roi


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

    filtered = cv.GaussianBlur(image, (3, 3), sigmaX=0, sigmaY=0)
    gray_im = cv.cvtColor(filtered, cv.COLOR_BGR2GRAY)
    er = cv.erode(gray_im, kernel=(5, 5), iterations=10)
    dil = cv.dilate(er, kernel=(5, 5), iterations=10)
    cann = cv.Canny(dil, threshold1=30, threshold2=130, L2gradient=True)
    final = cv.GaussianBlur(cann, (3, 3), sigmaX=0, sigmaY=0)

    circles2 = cv.HoughCircles(dil, cv1.CV_HOUGH_GRADIENT, dp=1, minDist=final.shape[0]/4,
                              param1=130, param2=5, minRadius=5, maxRadius=15)

    circles2 = np.uint16(np.around(circles2))
    #print circles
    cimage2 = image.copy()

    for i in circles2[0,:]:
        # draw the outer circle
        cv.circle(cimage2, (i[0],i[1]), i[2], (0,0,255), 2)
        # draw the center of the circle
        cv.circle(cimage2, (i[0],i[1]), 2, (200,20,200), 3)

    cimage3 = image.copy()
    countour, hierarchy = cv.findContours(final, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
    for c in xrange(len(countour)):
        cv.drawContours(cimage3, contours=countour, contourIdx=c, color=(0,10*c,0), lineType=cv.CV_AA)

    cv.namedWindow("Original-Filtered", cv.WINDOW_NORMAL)
    a = np.hstack((image, filtered))
    cv.imshow("Original-Filtered", a)

    cv.namedWindow("Erode-Dilate\nCanny-Final", cv.WINDOW_NORMAL)
    b = np.hstack((er, dil))
    c = np.hstack((cann, final))
    d = np.vstack((b, c))
    cv.imshow("Erode-Dilate\nCanny-Final", d)

    cv.namedWindow("Circles1", cv.WINDOW_NORMAL)
    cv.imshow("Circles1", cimage)
    cv.namedWindow("Circles2", cv.WINDOW_NORMAL)
    cv.imshow("Circles2", cimage2)
    cv.namedWindow("Circles3", cv.WINDOW_NORMAL)
    cv.imshow("Circles3", cimage3)
    cv.waitKey(0)
    cv.destroyAllWindows()

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
        gray = cv.equalizeHist(gray)
        blur = cv.GaussianBlur(gray, (5, 5), sigmaX=0, sigmaY=0)
        #filtered = cv.addWeighted(gray, 1.5, blur, -0.5, 0)
        #gray = cv.medianBlur(gray, 7)
        #gray = sharpen(src2, show=False)
        circles = cv.HoughCircles(blur, cv1.CV_HOUGH_GRADIENT, dp=1, minDist=gray.shape[0]/2,
                              param1=130, param2=15, minRadius=5, maxRadius=20)
        '''
        Optimal parameters:
            minDist = rows/4
            param1 = 130
            param2 = 15
            minRadius = 8
            maxRadius = 15
        '''

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

        k = cv.waitKey(100) & 0XFF
        if k == 27:
            cv.destroyAllWindows()
            break


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

    def pyramidal_filter(original_image, size):
        #return cv.pyrMeanShiftFiltering(original_image, sp=3, sr=size, maxLevel=2)
        return color_image

    cv.namedWindow("Smoothing", cv.WINDOW_NORMAL)
    cv.createTrackbar("Kernel", "Smoothing", 0, 9, nothing)
    filter_type = "0: Blur\n1: Gaussian\n2: Median\n3: Bilateral\n4: Pyramidal"
    cv.createTrackbar(filter_type, "Smoothing", 0, 4, nothing)

    while True:
        f_type = cv.getTrackbarPos(filter_type, "Smoothing")
        kernel = cv.getTrackbarPos("Kernel", "Smoothing")
        if f_type == 0:
            im_filtered = blur_filter(image, kernel)
        elif f_type == 1:
            im_filtered = gaussian_filter(image, kernel)
        elif f_type == 2:
            im_filtered = median_filter(image, kernel)
        elif f_type == 3:
            im_filtered = bilateral_filter(image, kernel)
        else:
            im_filtered = pyramidal_filter(color_image, kernel)

        cv.imshow("Smoothing", im_filtered)

        k = cv.waitKey(1) & 0XFF
        if k == 27:
            cv.destroyAllWindows()
            break


def diff_frame(frame1, frame2):
    im1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    im2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    im1 = cv.GaussianBlur(im1, (3, 3), sigmaY=0, sigmaX=0)
    im2 = cv.GaussianBlur(im2, (3, 3), sigmaY=0, sigmaX=0)
    diff = cv.absdiff(im1, im2)
    kernel = np.ones((3,3), np.uint8)
    opening = cv.morphologyEx(diff, cv.MORPH_OPEN, kernel, iterations = 1)
    cv.namedWindow("Difference", cv.WINDOW_NORMAL)
    cv.namedWindow("Dilatation", cv.WINDOW_NORMAL)
    cv.imshow("Difference", diff)
    cv.imshow("Dilatation", opening)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return diff


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
    lapl = cv.Laplacian(fil, cv.CV_32F)

    n = 1.5*gray2 - 0.5*fil - 0.01*5*np.multiply(gray2, 2*lapl)
    n = cv.convertScaleAbs(n, cv.CV_8U)

    if show:
        cv.namedWindow("Normal", cv.WINDOW_NORMAL)
        cv.namedWindow("Sharp", cv.WINDOW_NORMAL)
        cv.namedWindow("WOW", cv.WINDOW_NORMAL)
        cv.imshow("Normal", image)
        cv.imshow("Sharp", sharp)
        cv.imshow("WOW", n)
        cv.waitKey(0)
        cv.destroyAllWindows()

    return n


def detect_background(frame1, frame2):
    f1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    f2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    f1 = cv.equalizeHist(f1)
    f2 = cv.equalizeHist(f2)
    f1 = cv.GaussianBlur(f1, (5, 5), 5)
    f2 = cv.GaussianBlur(f2, (5, 5), 5)
    diff = cv.absdiff(f1, f2)
    transform = cv.dilate(diff, kernel=(5, 5), iterations=10)
    transform = cv.erode(transform, kernel=(5, 5), iterations=10)
    ret, binary = cv.threshold(transform, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    closing = cv.morphologyEx(binary, kernel=(5, 5), op=cv.MORPH_DILATE, iterations=10)
    mask = cv.medianBlur(closing, 9)

    masked_im = cv.bitwise_and(frame1, frame1, mask=mask)

    cv.namedWindow("Original", cv.WINDOW_NORMAL)
    original = np.vstack((f1, f2))
    cv.imshow("Original", original)
    cv.namedWindow("Difference-Transformation", cv.WINDOW_NORMAL)
    t = np.hstack((diff, transform))
    cv.imshow("Difference-Transformation", t)
    cv.namedWindow("Otsu", cv.WINDOW_NORMAL)
    cv.imshow("Otsu", binary)
    cv.namedWindow("Closed Image and Mask", cv.WINDOW_NORMAL)
    masking = np.hstack((closing, mask))
    cv.imshow("Closed Image and Mask", masking)
    cv.namedWindow("Masked Image", cv.WINDOW_NORMAL)
    cv.imshow("Masked Image", masked_im)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return masked_im


def circles_by_contour(image, full_dislplay):
    im = cv.GaussianBlur(image, (5, 5), 0)

    x = np.array([0, 0])
    y = np.array([0, 0])
    #x[0], y[0], x[1], y[1] = roi.getROI(im)
    x[1] = image.shape[1]
    y[1] = image.shape[0]

    copy = im.copy()
    mask = np.zeros((image.shape[0], image.shape[1], 3))
    mask[y[0]:y[1], x[0]:x[1], :] = 1
    masked = cv.multiply(copy, mask, dtype=cv.CV_8U)
    cropped = im.copy()
    cropped = cropped[y[0]:y[1], x[0]:x[1]]

    color_mask = color.color_mask(cropped, color='b', display=full_dislplay)
    gray = cv.cvtColor(color_mask, cv.COLOR_HSV2BGR)
    #sharp = sharpen(gray, show=True)
    #equal = histogram_eq(gray)
    gray = cv.cvtColor(gray, cv.COLOR_BGR2GRAY)

    ret, thresh = cv.threshold(gray, 120, 255, cv.THRESH_BINARY)

    if full_dislplay:
        cv.namedWindow("HSVMask, Threshold", cv.WINDOW_NORMAL)
        cv.imshow("HSVMask, Threshold", np.hstack((gray, thresh)))
        cv.waitKey(0)
        cv.destroyAllWindows()

    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    i = 0
    index = 0
    biggest_area = 0
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area >= biggest_area:
            biggest_area = area
            index = i
        i += 1

    (x_c, y_c), radius = cv.minEnclosingCircle(contours[index])
    #Relative Circle
    center = (int(x_c), int(y_c))
    radius = int(radius)

    #Absolute Center
    center_abs = (int(x_c) + x[0], int(y_c) + y[0])

    cv.circle(copy, center_abs, radius, (10, 10, 255), 2)
    print "Center: ", center_abs
    cv.imshow("Marker", copy)
    #cv.waitKey(0)
    #cv.destroyAllWindows()


def main():
    print "VIRTUAL BOARD\n\n"
    cv.destroyAllWindows()

    dir_name = "images/images_azul/"
    files1 = sorted(np.array(glob.glob(dir_name + "c1_image*.png")))
    files2 = sorted(np.array(glob.glob(dir_name + "c2_image*.png")))

    cv.namedWindow("Marker", cv.WINDOW_NORMAL)
    total_frames = len(files1)
    frame_number = 0
    while frame_number < total_frames:
        circles_by_contour(cv.imread(files2[frame_number]), full_dislplay=False)
        k = cv.waitKey(10) & 0XFF
        if k == 27:
            cv.destroyAllWindows()
            break
        frame_number += 1

    src2 = cv.imread(files1[40])
    sph1 = cv.imread("images/blue_sphere_00.png")
    sph2 = cv.imread("images/blue_sphere_01.png")

    #detect_all_circles(dir_name)
    #detect_circles(src2)
    #filtering(src2)

    #color.histogram_hsv(sph1, sph2)
    #bp = color.back_projection(src2)
    #detect_circles(bp)
    #color.histogram_ycc(sph1)
    '''
    for f in files1:
        frame = cv.imread(f)
        #color.color_mask(frame)
    '''

    f1 = cv.imread(files1[40])
    f2 = cv.imread(files1[41])
    #diff_frame(f1, f2)
    #histogram_eq(f1)
    #sharpen(f1)
    #maskedIm = detect_background(f1, f2)
    #detect_circles(maskedIm)

if __name__ == "__main__":
    main()
    cv.destroyAllWindows()

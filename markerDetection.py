__author__ = 'laverden'


import numpy as np
import cv2 as cv
import glob
import Color as Color
import ROISelection as Cropper


def diff_frame(frame1, frame2, display):
    im1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    im2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    im1 = cv.GaussianBlur(im1, (3, 3), sigmaY=0, sigmaX=0)
    im2 = cv.GaussianBlur(im2, (3, 3), sigmaY=0, sigmaX=0)
    diff = cv.absdiff(im1, im2)
    kernel = np.ones((3,3), np.uint8)
    opening = cv.morphologyEx(diff, cv.MORPH_OPEN, kernel, iterations=2)

    if display:
        cv.namedWindow("Difference-OP", cv.WINDOW_NORMAL)
        cv.imshow("Difference-OP", np.hstack((diff, opening)))
        cv.waitKey(0)
        cv.destroyAllWindows()

    return opening


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


def circles_by_contour(image, c, roi, full_display):
    im = cv.GaussianBlur(image, (9, 9), 0)

    copy = im.copy()
    cropped = im.copy()
    cropped = cropped[roi[1]:roi[3], roi[0]:roi[2]]

    color_mask = Color.color_mask(cropped, color=c, display=full_display)
    gray = cv.cvtColor(color_mask, cv.COLOR_HSV2BGR)
    gray = cv.cvtColor(gray, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 3)
    #ret, thresh = cv.threshold(gray, 120, 255, cv.THRESH_BINARY)
    ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    eros = cv.erode(thresh, (7, 7), 30)
    #thresh = cv.dilate(eros, (5,5), 30)

    if full_display:
        cv.namedWindow("Erosion-Dilatation", cv.WINDOW_NORMAL)
        cv.imshow("Erosion-Dilatation", np.hstack((eros, thresh)))
        cv.waitKey(0)
        cv.destroyAllWindows()

    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    i = 0
    index = -1
    biggest_area = 25

    for cnt in contours:
        area = cv.contourArea(cnt)
        if biggest_area <= area:
            biggest_area = area
            index = i
        i += 1
    #print "[%s] = %d" % (i, biggest_area)
    if index >= 0:
        (x_c, y_c), radius = cv.minEnclosingCircle(contours[index])
    else:
        x_c = y_c = 0
        radius = 0
        #print "No CIRCLE"

    # Relative Circle
    center = (int(x_c), int(y_c))
    radius = int(radius)
    # Absolute Center
    center_abs = (int(x_c) + roi[0], int(y_c) + roi[1])

    if radius != 0:
        cv.circle(copy, center_abs, 1, (10, 255, 25), 5)
        cv.circle(copy, center_abs, radius, (10, 10, 255), 2)
        #print "Center: ", center_abs

    return copy


def main():
    print "VIRTUAL BOARD\n\n"

    dir_name = "images/images_verde/"
    color_ball = 'g'
    files1 = sorted(np.array(glob.glob(dir_name + "c1_image*.png")))
    files2 = sorted(np.array(glob.glob(dir_name + "c2_image*.png")))

    ###################################################################
    ## CONTOUR METHOD APPLYING COLOR MASK OVER ALL FRAMES
    ###################################################################
    files = files1
    total_frames = len(files)
    frame_number = 0

    cal_frame = cv.imread(files[0])
    # WHAT: roi_xy = [x1, y1, x2, y2]
    roi_xy = np.array([0, 0, cal_frame.shape[1], cal_frame.shape[0]])
    roi_xy[0], roi_xy[1], roi_xy[2], roi_xy[3] = Cropper.getROI(cal_frame)

    cv.namedWindow("Marker", cv.WINDOW_NORMAL)
    while frame_number < total_frames:
        c = circles_by_contour(cv.imread(files[frame_number]), c=color_ball, roi=roi_xy, full_display=False)
        cv.imshow("Marker", c)
        k = cv.waitKey(10) & 0XFF
        if k == 27:
            cv.destroyAllWindows()
            break
        frame_number += 1
    '''
    ###################################################################
    ## CONTOUR METHOD APPLYING COLOR MASK OVER ONE FRAME -> DEBUGGING
    ###################################################################
    c = circles_by_contour(cv.imread(files1[1]), c='b', roi=roi_xy, full_display=True)
    cv.namedWindow("Marker", cv.WINDOW_NORMAL)
    cv.imshow("Marker", c)
    cv.waitKey(0)
    cv.destroyAllWindows()


    #color.histogram_hsv(sph1, sph2)
    #bp = color.back_projection(src2)
    #detect_circles(bp)
    #color.histogram_ycc(sph1)

    #histogram_eq(f1)
    #sharpen(f1)
    #maskedIm = detect_background(f1, f2)
    #detect_circles(maskedIm)

    ###################################################################
    ## CONTOUR METHOD USING BACKGROUND SEGMENTATION AND COLOR MASK
    ## NEED THE BACKGROUND MODEL
    f1 = cv.imread(files1[40])
    f2 = cv.imread(files1[41])
    hsv1 = Color.color_mask(f1, 'r', display=False)
    hsv2 = Color.color_mask(f2, 'r', display=False)
    hsv1 = cv.cvtColor(hsv1, cv.COLOR_HSV2BGR)
    hsv2 = cv.cvtColor(hsv2, cv.COLOR_HSV2BGR)
    diff = diff_frame(hsv1, hsv2, display=False)
    ret, thresh = cv.threshold(diff, 50, 255, cv.THRESH_BINARY)
    cv.namedWindow("BLA", cv.WINDOW_NORMAL)
    cv.imshow("BLA", thresh)
    cv.waitKey(0)

    copy = f1.copy()
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    i = 0
    index = -1
    biggest_area = 25
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area >= biggest_area and area < 1000:
            biggest_area = area
            index = i
        i += 1
    print "[%s] = %d" % (i, biggest_area)
    if index >= 0:
        (x_c, y_c), radius = cv.minEnclosingCircle(contours[index])
    else:
        x_c = y_c = 0
        radius = 0
        print "No CIRCLE"
    #Relative Circle
    center = (int(x_c), int(y_c))
    radius = int(radius)

    cv.circle(copy, center, 1, (10, 255, 25), 5)
    cv.circle(copy, center, radius, (10, 10, 255), 2)
    cv.namedWindow("MARKER", cv.WINDOW_NORMAL)
    cv.imshow("MARKER", copy)
    cv.waitKey(0)
    '''


if __name__ == "__main__":
    main()
    cv.destroyAllWindows()

# importing modules

import cv2
import numpy as np

# capturing video through webcam
cap = cv2.VideoCapture(0)

#
#
# lower = {'red':(166, 84, 141), 'green':(66, 122, 129), 'blue':(97, 100, 117), 'yellow':(23, 59, 119), 'orange':(0, 50, 80)} #assign new item lower['blue'] = (93, 10, 0)
# upper = {'red':(186,255,255), 'green':(86,255,255), 'blue':(117,255,255), 'yellow':(54,255,255), 'orange':(20,255,255)}
# colors = {'red':(0,0,255), 'green':(0,255,0), 'blue':(255,0,0), 'yellow':(0, 255, 217), 'orange':(0,140,255)}
#
#

while (1):
    _, img = cap.read()

    # converting frame(img i.e BGR) to HSV (hue-saturation-value)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # definig the range of red color
    # red_lower = np.array([166, 84, 141], np.uint8)
    # red_upper = np.array([186, 255, 255], np.uint8)
    red_lower = np.array([0, 100, 100], np.uint8)
    red_upper = np.array([5, 255, 255], np.uint8)
    # lower_red = np.array([0, 50, 50])
    # upper_red = np.array([10, 255, 255])

    # defining the Range of Blue color
    blue_lower = np.array([97, 100, 117], np.uint8)
    blue_upper = np.array([117, 255, 255], np.uint8)

    # defining the Range of yellow color
    yellow_lower = np.array([23, 59, 119], np.uint8)
    yellow_upper = np.array([54, 255, 255], np.uint8)

    green_lower = np.array([50, 50, 50], np.uint8)
    green_upper = np.array([70, 255, 255], np.uint8)

    orange_lower = np.array([5, 50, 50], np.uint8)
    orange_upper = np.array([15, 255, 255], np.uint8)

    # finding the range of red,blue and yellow color in the image
    red = cv2.inRange(hsv, red_lower, red_upper)
    blue = cv2.inRange(hsv, blue_lower, blue_upper)
    yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
    green = cv2.inRange(hsv, green_lower, green_upper)
    orange = cv2.inRange(hsv, orange_lower, orange_upper)

    # Morphological transformation, Dilation
    kernal = np.ones((5, 5), "uint8")

    red = cv2.dilate(red, kernal)
    res = cv2.bitwise_and(img, img, mask=red)

    blue = cv2.dilate(blue, kernal)
    res1 = cv2.bitwise_and(img, img, mask=blue)

    yellow = cv2.dilate(yellow, kernal)
    res2 = cv2.bitwise_and(img, img, mask=yellow)

    green = cv2.dilate(green, kernal)
    res3 = cv2.bitwise_and(img, img, mask=green)

    orange = cv2.dilate(orange, kernal)
    res4 = cv2.bitwise_and(img, img, mask=orange)

    # Tracking the Red Color
    (_, contours, hierarchy) = cv2.findContours(red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 2000):
            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(img, "Red color", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255))
    # cv2.imshow("Red", res)

    # Tracking the Blue Color
    (_, contours, hierarchy) = cv2.findContours(blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 2000):
            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "Blue color", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0))

    # Tracking the yellow Color
    (_, contours, hierarchy) = cv2.findContours(yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 2000):
            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.putText(img, "yellow  color", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0))

            # Tracking the yellow Color
    (_, contours, hierarchy) = cv2.findContours(green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 2000):
            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "green  color", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))

    (_, contours, hierarchy) = cv2.findContours(orange, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # for pic, contour in enumerate(contours):
    #     area = cv2.contourArea(contour)
    #     if (area > 300):
    #         x, y, w, h = cv2.boundingRect(contour)
    #         img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #         cv2.putText(img, "orange  color", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
    if len(contours) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        print 'center: ', center

        # only proceed if the radius meets a minimum size. Correct this value for your obect's size
        if radius > 0.5:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(img, (int(x), int(y)), int(radius), (0,0,0), 2)
            cv2.putText(img, "ball", (int(x - radius), int(y - radius)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 0, 0), 2)


    cv2.imshow("Orange", res4)




            # cv2.imshow("Redcolour",red)
    cv2.imshow("Color Tracking", img)
    # cv2.imshow("red",res)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break



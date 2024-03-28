import numpy as np
import cv2
import time

vid = cv2.VideoCapture("Road.mov")

while (True):
    ret, img = vid.read()

    # CROPPING IMAGE
    cropped_img = img[840:1920, 0:1080]

    # MAKING YELLOW BRIGHTER
    lowyellow = np.array([50, 100, 150])
    lightyellow = np.array([120, 215, 240])
    yellow = cv2.inRange(img, lowyellow, lightyellow)

    # HIGHLIGHTING YELLOW
    highlighted = cv2.cvtColor(yellow, cv2.COLOR_GRAY2BGR) # converting it to color so arrays are same size
    highlighted_img = cv2.add(img, highlighted)

    # PERSPECTIVE TRANSFORM
    pts1 = np.float32([[480, 1250], [620, 1250], [150, 1550], [950, 1550]])
    pts2 = np.float32([[0, 0], [475, 0], [0, 640], [475, 640]])

    # PERSPECTIVE WARP
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(highlighted_img, matrix, (475, 640))
    result_for_adding = cv2.warpPerspective(highlighted_img, matrix, (475, 640)) # black background for adding
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 25, 175)

    # MAKING BLACK FRAME FOR LINE OVERLAY
    cv2.circle(result_for_adding, (200,400), 250, (0,0,0), 500)

    # CRUDE LINE DETECTION
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=50, minLineLength=5, maxLineGap=250)

    if lines is not None:

        right_lines = []
        left_lines = []
        slopes_left = []
        slopes_right = []
        valid_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            slope = (y2-y1) / ((x2-x1) + 1) # +1 prevents denominator from being 0
            abs_slope = abs(slope) + 0.01 # +.01 to prevent slope from being inf
            slope_degrees = np.rad2deg(np.arctan(slope))

            if slope_degrees > 75 or slope_degrees < -75:
                valid_lines.append(line)

        for line in valid_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 5)

            if x1 > 0 and x1 < 150:
                left_lines.append(line)
            if x1 > 325 and x1 < 475:
                right_lines.append(line)

        for line in right_lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / ((x2 - x1) + 1)
            slopes_right.append(slope)

        for line in left_lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / ((x2 - x1) + 1)
            slopes_left.append(slope)

        avg_slopes_left = sum(slopes_left) / (len(slopes_left) + 0.01)
        avg_slopes_right = sum(slopes_right) / (len(slopes_right) + 0.01)

        print(np.rad2deg(np.arctan(avg_slopes_left)))

            # x1_true = max(x1, x2) # making x1 and y1 always on bottom
            # x2_true = min(x1, x2)
            # y1_true = max(y1, y2)
            # y2_true = min(y1, y2)
            #
            # bottom_y_value = y1_true
            #
            # #cv2.line(result_for_adding, (x1, y1), (x2, y2), (0, 0, 255), 5)
            # cv2.line(result, (x1_true, y1_true), (x2_true, y2_true), (0, 0, 255), 5)
            # cv2.circle(result, (x1_true, y1_true), 3, (255, 255, 255), 5)

            # if x1_true < 150 and x1_true > 0:
            #     left_lines.append(line)
            # if x1_true < 475 and x1_true > 325:
            #     right_lines.append(line)
            # else:
            #     pass
            #
            # for line in left_lines:
            #     lx1, ly1, lx2, ly2 = line[0]
            #     #bottom_x
            #     slope_left = (ly2-ly1) / ((lx2-lx1) + 1)
            #     slopes_left.append(slope_left)
            #
            # for line in right_lines:
            #     rx1, ry1, rx2, ry2 = line[0]
            #     slope_right = (ry2-ry1) / ((rx2-rx1) + 1)
            #     slopes_right.append(slope_right)
            #
            # avg_slope_left = sum(slopes_left) / (len(slopes_left) + .01)
            # avg_slope_right = sum(slopes_right) / (len(slopes_right) + .01)


        # EXTEND LINES TO TOP AND BOTTOM OF FRAME
        # GET X1 AND X2 FOR ALL LINES ON BOTH SIDES
        # CHECK IF X's ARE WITHIN CERTAIN VALUES THAT ARE ACCEPTABLE AS A LANE (0 -> 100 on left, 400 -> 500 on right)
        # IF TRUE
        #   GET AVERAGE X1 AND X2
        #   DRAW LINES BY USING TOP AND BOTTOM AS Y VALUES THEN AVERAGE X's FOR X VALUES


    # UN-PERSPECTIVE TRANSFORM
    matrix_reverse = cv2.getPerspectiveTransform(pts2, pts1)
    result_reverse = cv2.warpPerspective(result_for_adding, matrix_reverse, (1080, 1920))

    # CROPPING RESULT
    cropped_result_reverse = result_reverse[840:1920, 0:1080]

    # ADDING IMAGES
    added_overlay = cv2.add(cropped_img, cropped_result_reverse)
    gray_added_images = cv2.cvtColor(added_overlay, cv2.COLOR_BGR2GRAY)

    # PERSON DETECTION
    person_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
    people = person_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in people:
        cv2.rectangle(added_overlay, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # SHOW IMAGE
    cv2.imshow('added_overlay', added_overlay)
    cv2.imshow('result', result)

    #time.sleep(0.02)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()

# Avergae function https://www.geeksforgeeks.org/find-average-list-python/

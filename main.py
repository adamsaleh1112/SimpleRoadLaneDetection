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
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=50, minLineLength=5, maxLineGap=150)

    if lines is not None:

        right_lines = []
        left_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            slope = (y2-y1) / ((x2-x1) + 1) # +1 prevents denominator from being 0
            abs_slope = abs(slope) + 0.01

            # if slopes are positive, x1 y1 is on top

            x1_true = max(x1, x2)
            x2_true = min(x1, x2)
            y1_true = max(y1, y2)
            y2_true = min(y1, y2)

            bottom_y_value = y1_true
            distance_to_bottom = 640 - bottom_y_value
            extended_x_value = int(x1_true + (x1_true/abs_slope))
            extended_y_value = 640

            #cv2.line(result_for_adding, (x1, y1), (x2, y2), (0, 0, 255), 5)
            cv2.line(result, (extended_x_value, extended_y_value), (x2_true, y2_true), (0, 0, 255), 5)
            cv2.circle(result, (x1_true, y1_true), 3, (255, 255, 255), 5)


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

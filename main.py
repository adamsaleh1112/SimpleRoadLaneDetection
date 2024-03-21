import numpy as np
import cv2
import time

vid = cv2.VideoCapture("Road.mov")

while (True):
    ret, img = vid.read()

    cropped_img = img[840:1920, 0:1080]

    # PERSPECTIVE TRANSFORM
    pts1 = np.float32([[480, 1250], [620, 1250], [150, 1550], [950, 1550]])
    pts2 = np.float32([[0, 0], [475, 0], [0, 640], [475, 640]])

    # PERSPECTIVE WARP
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, matrix, (475, 640))
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # CRUDE LINE DETECTION
    lines = cv2.HoughLinesP(edges, rho=10, theta=np.pi / 180, threshold=50, minLineLength=5, maxLineGap=5)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2-y1) / (x2-x1)
            arctan_slope = np.arctan(slope)
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 5)

    # UN-PERSPECTIVE TRANSFORM
    matrix_reverse = cv2.getPerspectiveTransform(pts2, pts1)
    result_reverse = cv2.warpPerspective(result, matrix_reverse, (1080, 1920))

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
    cv2.imshow('frame', added_overlay)

    time.sleep(0.02)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # розмиття для зменшення шуму
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # діапазон кольору (ціль - червоний)
    lower = np.array([0, 120, 70])
    upper = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # пошук контурів
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    frame_h, frame_w = frame.shape[:2]
    center_x, center_y = frame_w // 2, frame_h // 2

    # хрестик в центрі кадру
    cv2.drawMarker(frame, (center_x, center_y), (255, 255, 0), cv2.MARKER_CROSS, 20, 3)

    if contours:
        # найбільший контур
        c = max(contours, key=cv2.contourArea)

        if cv2.contourArea(c) > 500:  # фільтр шуму
            M = cv2.moments(c)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # похибка від центру кадру
            error_x = cx - center_x
            error_y = cy - center_y









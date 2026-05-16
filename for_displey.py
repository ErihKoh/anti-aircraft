import cv2
import numpy as np
import serial

arduino = serial.Serial('/dev/tty.wchusbserial10', 9600, timeout=1)

def detect_color(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    blue_lower = np.array([100, 80, 40])
    blue_upper = np.array([140, 255, 255])
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

    red_lower = np.array([0, 120, 70])
    red_upper = np.array([10, 255, 255])
    red_mask = cv2.inRange(hsv, red_lower, red_upper)

    for mask, color_name in [(blue_mask, "BLUE"), (red_mask, "RED")]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > 3000:
                return color_name, c

    return "NONE", None

cap = cv2.VideoCapture(0)
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
center_x, center_y = frame_w // 2, frame_h // 2

last_msg = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Хрестик центру кадру
    cv2.drawMarker(frame, (center_x, center_y), (255, 255, 0), cv2.MARKER_CROSS, 20, 2)

    color, contour = detect_color(frame)

    if contour is not None:
        # Межі обʼєкта
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

        # Центр обʼєкта
        M = cv2.moments(contour)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Хрестик на обʼєкті
        cv2.drawMarker(frame, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)

        # Лінія від центру кадру до обʼєкта
        cv2.line(frame, (center_x, center_y), (cx, cy), (0, 255, 255), 1)

        # Координати на екрані
        cv2.putText(frame, f"{color} X:{cx} Y:{cy}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Відправка на дисплей
        line1 = f"{color} X:{cx:<5}"[:16]
        line2 = f"Y:{cy:<5}"[:16]
        msg = f"{line1},{line2}\n"

        if msg != last_msg:
            arduino.write(msg.encode())
            last_msg = msg
    else:
        cv2.putText(frame, "No target", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        msg = "No target       ,                \n"
        if msg != last_msg:
            arduino.write(msg.encode())
            last_msg = msg

    cv2.imshow("Color Tracking", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
arduino.close()
cv2.destroyAllWindows()
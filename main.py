import cv2
import numpy as np


def error_to_servo(error, frame_size):
    normalized = np.clip(error / (frame_size / 2), -1, 1)
    return 90 + normalized * 90


def main():
    cap = cv2.VideoCapture(0)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    center_x, center_y = frame_w // 2, frame_h // 2

    servo_x = 90.0
    servo_y = 90.0

    lower = np.array([90, 80, 40])
    upper = np.array([140, 255, 255])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawMarker(frame, (center_x, center_y), (255, 255, 0), cv2.MARKER_CROSS, 20, 3)

        if contours:
            c = max(contours, key=cv2.contourArea)

            if cv2.contourArea(c) > 500:
                M = cv2.moments(c)
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                error_x = cx - center_x
                error_y = cy - center_y

                servo_x = error_to_servo(error_x, frame_w)
                servo_y = error_to_servo(error_y, frame_h)

                cv2.drawMarker(frame, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                cv2.drawContours(frame, [c], -1, (0, 255, 0), 2)

                cv2.putText(frame, f"Error X: {error_x:+d}  Servo X: {servo_x:.1f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                cv2.putText(frame, f"Error Y: {error_y:+d}  Servo Y: {servo_y:.1f}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Target not found", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Target not found", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        cv2.imshow("Tracking", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
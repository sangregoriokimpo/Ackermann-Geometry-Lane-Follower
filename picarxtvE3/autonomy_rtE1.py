# autonomy_rt.py
import cv2
import numpy as np
import time
from picarx import Picarx
from sign_detection import SignDetector

# 🛣️ Improved lane-following logic
def detect_lane_angle(frame):
    h, w = frame.shape[:2]
    roi = frame[h//2:, :]  # Bottom half
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=20, maxLineGap=300)
    left_lines, right_lines = [], []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            if slope < -0.5:
                left_lines.append((x1, y1, x2, y2))
            elif slope > 0.5:
                right_lines.append((x1, y1, x2, y2))

    road_center = None
    frame_center = w // 2
    if left_lines and right_lines:
        left_x = np.mean([x1 for x1, _, x2, _ in left_lines])
        right_x = np.mean([x2 for x1, _, x2, _ in right_lines])
        road_center = int((left_x + right_x) / 2)
    elif left_lines:
        left_x = np.mean([x1 for x1, _, _, _ in left_lines])
        road_center = int(left_x + 80)
    elif right_lines:
        right_x = np.mean([x2 for _, _, x2, _ in right_lines])
        road_center = int(right_x - 80)

    angle = 0
    if road_center is not None:
        error = road_center - frame_center
        normalized_error = error / frame_center
        angle = int(normalized_error * 30)  # Max ±30°
        angle = np.clip(angle, -30, 30)

    return angle, edges, roi


# 🚗 Init car and sign detection
car = Picarx()
sign_detector = SignDetector()

cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 240)

frame_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 🛣️ Lane-following angle
        steering_angle, lane_edges, lane_roi = detect_lane_angle(frame)

        # 🧭 Apply steering
        car.set_dir_servo_angle(steering_angle)
        car.forward(20)

        # 🚦 Sign detection async every few frames
        if frame_count % 5 == 0:
            sign_roi = frame[:120, :]  # top half
            sign_detector.detect_async(sign_roi)

        sign_label = ""
        result = sign_detector.get_result()
        if result == 11:
            sign_label = "🛑 Stop Sign"
            car.stop()
            time.sleep(2)
            sign_detector.detection_result = None
        elif result == 9:
            sign_label = "🚦 Traffic Light"

        # 🖼️ Visual overlay
        overlay = frame.copy()
        cv2.line(overlay, (160, 120), (160 + steering_angle * 2, 240), (255, 0, 255), 2)
        cv2.putText(overlay, f"Steering Angle: {steering_angle}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if sign_label:
            cv2.putText(overlay, f"Sign: {sign_label}", (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 🪟 Show lanes and signs
        cv2.imshow("Camera Feed", overlay)
        cv2.imshow("Lane ROI", lane_roi)
        cv2.imshow("Lane Edges", lane_edges)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

except KeyboardInterrupt:
    print("🛑 Interrupted")

finally:
    cap.release()
    car.stop()
    cv2.destroyAllWindows()

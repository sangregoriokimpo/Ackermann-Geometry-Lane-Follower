import cv2
import numpy as np
from picarx import Picarx
from time import sleep
from kalmanFilter import KalmanFilter


# -------- Vision Module ---------
class AckermannVision:
    def __init__(self, camera_device='/dev/video0'):
        self.cap = cv2.VideoCapture(camera_device, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        if not self.cap.isOpened():
            raise IOError(f"Cannot open camera {camera_device}")

        self.minAngle = -30
        self.maxAngle = 30

    def release(self):
        if self.cap.isOpened():
            self.cap.release()

    def region_selection(self, image):
        mask = np.zeros_like(image)
        rows, cols = image.shape[:2]
        vertices = np.array([[
            [cols * 0.1, rows * 0.95],
            [cols * 0.4, rows * 0.6],
            [cols * 0.6, rows * 0.6],
            [cols * 0.9, rows * 0.95]
        ]], dtype=np.int32)
        cv2.fillPoly(mask, vertices, 255)
        return cv2.bitwise_and(image, mask)

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None, None, None

        # Resize and center-crop
        frame = cv2.resize(frame, (320, 240))
        crop_width = 160
        center_x = frame.shape[1] // 2
        half_crop = crop_width // 2
        frame = frame[:, center_x - half_crop : center_x + half_crop]

        # Edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        masked = self.region_selection(edges)

        height, width = masked.shape
        scan_y = height - 5  # Row near the bottom

        # Find leftmost and rightmost edge pixels in the scan line
        scan_line = masked[scan_y]
        edge_indices = np.where(scan_line > 0)[0]

        if len(edge_indices) < 2:
            return None, frame.copy(), frame[int(240 * 0.4):] / 255.0  # Not enough edges

        left_x = edge_indices[0]
        right_x = edge_indices[-1]

        # Compute road center between left and right lines
        road_center = (left_x + right_x) // 2
        image_center = width // 2
        error = road_center - image_center

        # Normalize error and convert to angle
        steering_value = error / image_center
        steering_value = np.clip(steering_value, -1.0, 1.0)
        angle = ((steering_value + 1) / 2) * (self.maxAngle - self.minAngle) + self.minAngle

        # Visualization
        lane_img = cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR)
        cv2.circle(lane_img, (left_x, scan_y), 5, (255, 0, 0), -1)
        cv2.circle(lane_img, (right_x, scan_y), 5, (0, 0, 255), -1)
        cv2.circle(lane_img, (road_center, scan_y), 5, (0, 255, 0), -1)
        cv2.line(lane_img, (image_center, scan_y - 10), (image_center, scan_y + 10), (255, 255, 255), 1)

        cropped_input = frame[int(240 * 0.4):] / 255.0
        return angle, lane_img, cropped_input



# -------- PiCarX Controller --------
class PicarXController:
    def __init__(self):
        self.px = Picarx()
        self.vision = AckermannVision()
        self.kalman = KalmanFilter()
        self.mode = "manual"
        self.speed = 0.1

    def shutdown(self):
        print("Shutting down...")
        self.px.stop()
        self.vision.release()
        sleep(0.2)
        exit()

    def run(self):
        print("""
Modes:
  w/a/d/s - Manual drive
  space   - Stop
  v       - Vision lane follow (based on edge mask)
  q       - Quit
""")
        try:
            while True:
                angle, lane_img, _ = self.vision.process_frame()
                if lane_img is not None:
                    cv2.imshow("Lane View", lane_img)

                # Kalman smoothing
                smoothed_angle = self.kalman.update(angle) if angle is not None else None

                if self.mode == "vision":
                    if smoothed_angle is not None:
                        print(f"[Vision] Steering angle: {smoothed_angle:.2f}")
                        self.px.set_dir_servo_angle(int(smoothed_angle))
                        self.px.forward(self.speed)
                    else:
                        print("[Vision] No edge detected. Stopping.")
                        self.px.forward(0)

                # Manual driving controls
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.shutdown()
                elif key == ord(' '):
                    self.px.forward(0)
                    self.mode = "manual"
                elif key == ord('w'):
                    self.px.set_dir_servo_angle(0)
                    self.px.forward(self.speed)
                    self.mode = "manual"
                elif key == ord('a'):
                    self.px.set_dir_servo_angle(-30)
                    self.px.forward(self.speed)
                    self.mode = "manual"
                elif key == ord('d'):
                    self.px.set_dir_servo_angle(30)
                    self.px.forward(self.speed)
                    self.mode = "manual"
                elif key == ord('s'):
                    self.px.backward(self.speed)
                    self.mode = "manual"
                elif key == ord('v'):
                    self.kalman = KalmanFilter()  # Reset smoothing
                    self.mode = "vision"
                    print("Mode: VISION (Edge mask following)")
        finally:
            self.shutdown()


if __name__ == "__main__":
    controller = PicarXController()
    controller.run()

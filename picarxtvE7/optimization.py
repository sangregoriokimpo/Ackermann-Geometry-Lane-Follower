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

    def hough_transform(self, image):
        return cv2.HoughLinesP(image, 1, np.pi / 180, 20, minLineLength=20, maxLineGap=500)

    def average_slope_intercept(self, lines):
        left_lines, right_lines, left_weights, right_weights = [], [], [], []
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x1 == x2:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                length = np.hypot(x2 - x1, y2 - y1)
                if slope < 0:
                    left_lines.append((slope, intercept))
                    left_weights.append(length)
                else:
                    right_lines.append((slope, intercept))
                    right_weights.append(length)
        left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if left_weights else None
        right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if right_weights else None
        return left_lane, right_lane

    def pixel_points(self, y1, y2, line):
        if line is None:
            return None
        slope, intercept = line
        if abs(slope) < 0.01:
            return None
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return ((x1, int(y1)), (x2, int(y2)))

    def lane_lines(self, image, lines):
        left_lane, right_lane = self.average_slope_intercept(lines)
        y1 = image.shape[0]
        y2 = int(y1 * 0.6)
        return self.pixel_points(y1, y2, left_lane), self.pixel_points(y1, y2, right_lane)

    def draw_lane_lines(self, image, lines, road_center=None):
        output = image.copy()
        for line in lines:
            if line is not None:
                cv2.line(output, *line, (255, 0, 0), 6)
        if road_center is not None:
            cv2.circle(output, (road_center, image.shape[0] - 20), 5, (0, 255, 0), -1)
        return output

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None, None

        # Resize and crop horizontally to center
        frame = cv2.resize(frame, (320, 240))
        crop_width = 180
        center_x = frame.shape[1] // 2
        half_crop = crop_width // 2
        frame = frame[:, center_x - half_crop : center_x + half_crop]  # (240, 160, 3)

        # Crop vertically (keep bottom 60%)
        frame = frame[int(240 * 0.4):]  # now frame is (144, 160, 3)

        # Color filtering to isolate white and yellow lanes
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, (0, 0, 150), (180, 60, 255))
        yellow_mask = cv2.inRange(hsv, (15, 100, 100), (35, 255, 255))
        lane_mask = cv2.bitwise_or(white_mask, yellow_mask)  

        # Fallback brightness mask from grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Combine with brightness mask
        lane_mask = cv2.bitwise_or(lane_mask, bright_mask)  
        cv2.imshow("White/Yellow Mask", lane_mask)



        # Apply mask
        masked_frame = cv2.bitwise_and(frame, frame, mask=lane_mask)

        # Grayscale + blur
        gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Edge detection
        edges = cv2.Canny(blur, 50, 150)

        # Morphological closing to fill gaps
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Region of interest
        masked = self.region_selection(edges)

        # Line detection
        lines = self.hough_transform(masked)

        angle = None
        road_center = None
        if lines is not None:
            left_line, right_line = self.lane_lines(frame, lines)
            mid_x = frame.shape[1] // 2

            if left_line and right_line:
                road_center = (left_line[1][0] + right_line[1][0]) // 2
            elif left_line:
                road_center = left_line[1][0] + 80
            elif right_line:
                road_center = right_line[1][0] - 80

            if road_center is not None:
                error = road_center - mid_x
                steering_value = error / mid_x
                steering_value = np.clip(steering_value, -1.0, 1.0)
                angle = ((steering_value + 1) / 2) * (self.maxAngle - self.minAngle) + self.minAngle

            lane_img = self.draw_lane_lines(frame, [left_line, right_line], road_center)
        else:
            lane_img = frame.copy()

        return angle, lane_img



# -------- PiCarX Controller --------
class PicarXController:
    def __init__(self):
        self.px = Picarx()
        self.vision = AckermannVision()
        self.kalman = KalmanFilter()
        self.mode = "manual"  # options: manual, vision
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
  v       - Vision lane follow (with Kalman smoothing)
  q       - Quit
""")
        try:
            while True:
                angle, lane_img = self.vision.process_frame()
                if lane_img is not None:
                    cv2.imshow("Lane View", lane_img)

                if angle is not None:
                    smoothed_angle = self.kalman.update(angle)
                else:
                    smoothed_angle = None

                if self.mode == "vision" and smoothed_angle is not None:
                    self.px.set_dir_servo_angle(int(smoothed_angle))
                    self.px.forward(self.speed)

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
                    self.kalman = KalmanFilter()
                    self.mode = "vision"
                    print("Mode: VISION")
        finally:
            self.shutdown()


if __name__ == "__main__":
    controller = PicarXController()
    controller.run()

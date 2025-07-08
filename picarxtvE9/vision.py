import cv2
import numpy as np

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
            [cols * 0.2, rows * 0.9],
            [cols * 0.2, rows * 0.1],
            [cols * 0.8, rows * 0.1],
            [cols * 0.8, rows * 0.9]
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
        # Skip invalid slopes
        if abs(slope) < 1e-3 or np.isinf(slope) or np.isnan(slope):
            return None
        try:
            x1 = int((y1 - intercept) / slope)
            x2 = int((y2 - intercept) / slope)
            return ((x1, int(y1)), (x2, int(y2)))
        except ZeroDivisionError:
            return None


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
            print("⚠️ Warning: Failed to read camera frame.")
            return None, None, None

        frame = cv2.resize(frame, (320, 240))
        height = frame.shape[0]
        half = height // 2
        top_half = frame[:half, :]
        bottom_half = frame[half:, :]

        gray = cv2.cvtColor(bottom_half, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        masked = self.region_selection(edges)

        cv2.imshow("Masked ROI", masked)  # Debug output

        lines = self.hough_transform(masked)
        angle = None
        road_center = None

        if lines is not None:
            left_line, right_line = self.lane_lines(bottom_half, lines)
            mid_x = bottom_half.shape[1] // 2

            if left_line and right_line:
                road_center = (left_line[1][0] + right_line[1][0]) // 2
            elif left_line:
                road_center = left_line[1][0] + 100
            elif right_line:
                road_center = right_line[1][0] - 100

            if road_center is not None:
                error = road_center - mid_x
                steering_value = error / mid_x
                steering_value = np.clip(steering_value, -1.0, 1.0)
                angle = ((steering_value + 1) / 2) * (self.maxAngle - self.minAngle) + self.minAngle

            lane_img = self.draw_lane_lines(bottom_half, [left_line, right_line], road_center)
        else:
            lane_img = bottom_half.copy()

        return angle, lane_img, top_half

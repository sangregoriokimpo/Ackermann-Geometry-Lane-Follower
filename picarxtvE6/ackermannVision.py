import cv2
import numpy as np
import math

class AckermannVision:
    def __init__(self, cap):
        self.cap = cap
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("Camera not provided or failed to open.")

    def release(self):
        pass  # Camera released by controller

    def detect_edges(self, frame):
        # --- White line detection (HSV) ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 200], dtype="uint8")
        upper_white = np.array([180, 50, 255], dtype="uint8")
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)

        # --- Black lane detection (grayscale inverse) ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, black_mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)

        # --- Edge detection on white mask only ---
        edges = cv2.Canny(white_mask, 50, 150)

        return edges, white_mask, black_mask

    def region_of_interest(self, edges):
        height, width = edges.shape
        mask = np.zeros_like(edges)

        polygon = np.array([[
            (0, height),
            (0, int(height * 0.33)),
            (width, int(height * 0.33)),
            (width, height)
        ]], dtype=np.int32)

        cv2.fillPoly(mask, polygon, 255)
        return cv2.bitwise_and(edges, mask)


    def detect_line_segments(self, edges):
        return cv2.HoughLinesP(edges, 1, np.pi/180, 10, np.array([]),
                               minLineLength=10, maxLineGap=50)

    def average_slope_intercept(self, frame, line_segments):
        if line_segments is None:
            return []

        height, width, _ = frame.shape
        left_fit, right_fit = [], []

        boundary = 1 / 3
        left_boundary = width * (1 - boundary)
        right_boundary = width * boundary

        for segment in line_segments:
            for x1, y1, x2, y2 in segment:
                if x1 == x2:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1

                if slope < 0 and x1 < left_boundary and x2 < left_boundary:
                    left_fit.append((slope, intercept))
                elif slope > 0 and x1 > right_boundary and x2 > right_boundary:
                    right_fit.append((slope, intercept))

        lane_lines = []
        if left_fit:
            lane_lines.append(self.make_points(frame, np.mean(left_fit, axis=0)))
        if right_fit:
            lane_lines.append(self.make_points(frame, np.mean(right_fit, axis=0)))
        return lane_lines

    def make_points(self, frame, line):
        height, width, _ = frame.shape
        slope, intercept = line
        y1 = height
        y2 = int(height * 0.5)
        if slope == 0:
            slope = 0.1

        x1 = int(np.clip((y1 - intercept) / slope, 0, width - 1))
        x2 = int(np.clip((y2 - intercept) / slope, 0, width - 1))
        return [[x1, y1, x2, y2]]

    def get_steering_angle(self, frame, lane_lines):
        height, width, _ = frame.shape

        if len(lane_lines) == 2:
            _, _, left_x2, _ = lane_lines[0][0]
            _, _, right_x2, _ = lane_lines[1][0]
            mid = width // 2
            x_offset = (left_x2 + right_x2) // 2 - mid
        elif len(lane_lines) == 1:
            x1, _, x2, _ = lane_lines[0][0]
            x_offset = x2 - x1
        else:
            return None

        y_offset = height // 2
        if y_offset == 0:
            y_offset = 1
        angle_rad = math.atan(x_offset / y_offset)
        return math.degrees(angle_rad)

    def draw_lines(self, frame, lines, color=(0, 255, 0), thickness=4):
        overlay = np.zeros_like(frame)
        if lines:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(overlay, (x1, y1), (x2, y2), color, thickness)
        return cv2.addWeighted(frame, 0.8, overlay, 1, 1)

    def draw_heading(self, frame, angle, color=(0, 0, 255)):
        height, width, _ = frame.shape
        angle_rad = math.radians(angle)
        x1 = width // 2
        y1 = height
        x2 = int(x1 - height // 2 / math.tan(angle_rad))
        y2 = int(height * 0.5)
        cv2.line(frame, (x1, y1), (x2, y2), color, 3)
        return frame

    def estimate_black_lane_center(self, black_mask):
        contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width = black_mask.shape

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    x_offset = cx - width // 2
                    y_offset = height // 2
                    angle_rad = math.atan2(x_offset, y_offset)
                    return math.degrees(angle_rad)
        return None

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None, None, None

        # 👇 Crop bottom half
        height = frame.shape[0]
        frame = frame[height // 2:, :]

        edges, white_mask, black_mask = self.detect_edges(frame)
        roi = self.region_of_interest(edges)
        segments = self.detect_line_segments(roi)
        lanes = self.average_slope_intercept(frame, segments)
        angle = self.get_steering_angle(frame, lanes)

        # Fallback to black lane center
        if angle is None:
            fallback_angle = self.estimate_black_lane_center(black_mask)
            if fallback_angle is not None:
                print("[INFO] Using black lane center fallback")
                angle = fallback_angle

        # ✅ Only draw green lane lines (no red heading line)
        overlay = self.draw_lines(frame, lanes, color=(0, 255, 0), thickness=4)

        return angle, white_mask, overlay, black_mask
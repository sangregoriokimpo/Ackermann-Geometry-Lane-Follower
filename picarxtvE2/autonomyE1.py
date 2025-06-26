import cv2
import numpy as np
from ultralytics import YOLO
from picarx import Picarx
from time import sleep


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

        # Load YOLO model
        self.model = YOLO("/home/vehicle0/test_vehicle/best.pt")  # Replace with custom model path for street signs

    def get_rois(self, frame):
        height, width = frame.shape[:2]
        traffic_roi = frame[0:int(0.4 * height), :]
        lane_roi = frame[int(0.4 * height):, :]
        return traffic_roi, lane_roi

    def detect_signs(self, traffic_roi):
        results = self.model(traffic_roi, verbose=False)
        detections = []

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            detections.append({
                "class_id": cls_id,
                "confidence": conf,
                "bbox": tuple(xyxy)
            })
        return detections

    def region_selection(self, image):
        mask = np.zeros_like(image)
        ignore_mask_color = 255
        rows, cols = image.shape[:2]
        bottom_left = [cols * 0.1, rows * 0.95]
        top_left = [cols * 0.4, rows * 0.6]
        bottom_right = [cols * 0.9, rows * 0.95]
        top_right = [cols * 0.6, rows * 0.6]
        vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        return cv2.bitwise_and(image, mask)

    def hough_transform(self, image):
        return cv2.HoughLinesP(image, 1, np.pi / 180, 20, minLineLength=20, maxLineGap=500)

    def average_slope_intercept(self, lines):
        left_lines, right_lines = [], []
        left_weights, right_weights = [], []
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

    def pixel_points(self, y1, y2, line, slope_thresh=0.01):
        if line is None:
            return None
        slope, intercept = line
        if abs(slope) < slope_thresh:
            return None
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return ((x1, int(y1)), (x2, int(y2)))

    def lane_lines(self, image, lines):
        left_lane, right_lane = self.average_slope_intercept(lines)
        y1 = image.shape[0]
        y2 = y1 * 0.6
        left_line = self.pixel_points(y1, y2, left_lane)
        right_line = self.pixel_points(y1, y2, right_lane)
        return left_line, right_line

    def draw_lane_lines(self, image, lines, color=(255, 0, 0), thickness=6):
        line_image = np.zeros_like(image)
        for line in lines:
            if line is not None:
                cv2.line(line_image, *line, color, thickness)
        return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            print("Failed to grab frame")
            return None, None, None, None

        frame = cv2.resize(frame, (320, 240))
        traffic_roi, lane_roi = self.get_rois(frame)

        gray = cv2.cvtColor(lane_roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        masked = self.region_selection(edges)
        lines = self.hough_transform(masked)

        if lines is not None:
            left_line, right_line = self.lane_lines(lane_roi, lines)
            mid_x = lane_roi.shape[1] // 2

            if left_line and right_line:
                road_center = (left_line[1][0] + right_line[1][0]) // 2
            elif left_line:
                road_center = left_line[1][0] + 100
            elif right_line:
                road_center = right_line[1][0] - 100
            else:
                road_center = None

            if road_center is not None:
                error = road_center - mid_x
                steering_value = error / (lane_roi.shape[1] // 2)
                steering_value = max(-1.0, min(1.0, steering_value))
                angle_range = self.maxAngle - self.minAngle
                angle = (steering_value + 1) / 2 * angle_range + self.minAngle

                lane_image = self.draw_lane_lines(lane_roi, [left_line, right_line])
                cv2.circle(lane_image, (road_center, lane_roi.shape[0] - 20), 5, (0, 255, 0), -1)
                return angle, masked, lane_image, traffic_roi

        return None, masked, lane_roi, traffic_roi

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()


# -------- PiCar-X Controller ---------
class PicarXController:
    def __init__(self):
        self.px = Picarx()
        self.pan_angle = 0
        self.tilt_angle = 0
        self.speed = 10
        self.vision_mode = False
        self.vision = AckermannVision()

    def manual_control(self, key):
        if key == 'w':
            self.px.set_dir_servo_angle(0)
            self.px.forward(self.speed)
        elif key == 's':
            self.px.set_dir_servo_angle(0)
            self.px.backward(self.speed)
        elif key == 'a':
            self.px.set_dir_servo_angle(-30)
            self.px.forward(self.speed)
        elif key == 'd':
            self.px.set_dir_servo_angle(30)
            self.px.forward(self.speed)
        elif key == 'i':
            self.tilt_angle = min(self.tilt_angle + 5, 35)
        elif key == 'k':
            self.tilt_angle = max(self.tilt_angle - 5, -35)
        elif key == 'l':
            self.pan_angle = min(self.pan_angle + 5, 35)
        elif key == 'j':
            self.pan_angle = max(self.pan_angle - 5, -35)
        elif key == ' ':
            self.px.forward(0)
            self.px.set_dir_servo_angle(0)

        self.px.set_cam_tilt_angle(self.tilt_angle)
        self.px.set_cam_pan_angle(self.pan_angle)

    def vision_control(self):
        angle, mask, lane_output, traffic_roi = self.vision.process_frame()
        if lane_output is None:
            print("No frame!")
            return

        if angle is not None:
            steering_angle = int(angle)
            steering_angle = np.clip(steering_angle, -30, 30)
            print(f"[Vision] Steering Angle: {steering_angle}°")
            self.px.set_dir_servo_angle(steering_angle)
            self.px.forward(self.speed)
        else:
            print("Lane lost!")
            self.px.forward(0)

        if traffic_roi is not None:
            detections = self.vision.detect_signs(traffic_roi)
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                label = f"{det['class_id']} ({det['confidence']:.2f})"
                cv2.rectangle(traffic_roi, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(traffic_roi, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow("Traffic ROI", traffic_roi)

        cv2.imshow("Lane Detection", lane_output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.shutdown()

    def shutdown(self):
        print("Shutting down...")
        self.px.set_cam_tilt_angle(0)
        self.px.set_cam_pan_angle(0)
        self.px.set_dir_servo_angle(0)
        self.px.stop()
        self.vision.release()
        sleep(0.2)
        exit()

    def run(self):
        print("""
            Control keys:
                w/s/a/d: move
                i/k: tilt camera
                j/l: pan camera
                space: stop
                v: toggle vision mode
                q: quit
        """)
        try:
            while True:
                if self.vision_mode:
                    self.vision_control()
                else:
                    ret, frame = self.vision.cap.read()
                    if ret:
                        frame = cv2.resize(frame, (320, 240))
                        cv2.imshow("Camera View", frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('v'):
                        self.vision_mode = True
                    elif key == ord('q'):
                        break
                    elif key != 255:
                        self.manual_control(chr(key))

        finally:
            self.shutdown()


if __name__ == "__main__":
    controller = PicarXController()
    controller.run()

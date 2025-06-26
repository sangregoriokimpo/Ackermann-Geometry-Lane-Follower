import cv2
import numpy as np
import readchar
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
        if abs(slope) < slope_thresh:  # avoid divide-by-zero or near-horizontal lines
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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        masked = self.region_selection(edges)
        lines = self.hough_transform(masked)

        if lines is not None:
            left_line, right_line = self.lane_lines(frame, lines)
            mid_x = frame.shape[1] // 2

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
                steering_value = error / (frame.shape[1] // 2)
                steering_value = max(-1.0, min(1.0, steering_value))
                angle_range = self.maxAngle - self.minAngle
                angle = (steering_value + 1) / 2 * angle_range + self.minAngle

                lane_image = self.draw_lane_lines(frame, [left_line, right_line])
                cv2.circle(lane_image, (road_center, frame.shape[0] - 20), 5, (0, 255, 0), -1)
                return angle, masked, lane_image, None

        return None, masked, frame, None

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()


# -------- PiCar-X Controller ---------
class PicarXController:
    PAN_LIMIT = 35
    TILT_LIMIT = 35

    def __init__(self):
        print("Initializing PicarX controller...")
        self.px = Picarx()
        self.pan_angle = 0
        self.tilt_angle = 0
        self.speed = 0.1
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
            self.tilt_angle = min(self.tilt_angle + 5, self.TILT_LIMIT)
        elif key == 'k':
            self.tilt_angle = max(self.tilt_angle - 5, -self.TILT_LIMIT)
        elif key == 'l':
            self.pan_angle = min(self.pan_angle + 5, self.PAN_LIMIT)
        elif key == 'j':
            self.pan_angle = max(self.pan_angle - 5, -self.PAN_LIMIT)
        elif key == ' ':
            self.px.forward(0)
            self.px.set_dir_servo_angle(0)

        self.px.set_cam_tilt_angle(self.tilt_angle)
        self.px.set_cam_pan_angle(self.pan_angle)

    def vision_control(self):
        angle, mask, frame_out, lane_width_px = self.vision.process_frame()
        if frame_out is None:
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
            self.px.set_dir_angle(0)

        cv2.imshow("Vision", frame_out)
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
                w: forward
                s: backward
                a/d: left/right turn
                i/k: head tilt up/down
                j/l: head pan left/right
                v: toggle vision mode
                space: stop
                ctrl+c or q (in vision): quit
        """)
        try:
            while True:
                if self.vision_mode:
                    self.vision_control()
                else:
                    # --- Show camera ---
                    ret, frame = self.vision.cap.read()
                    if ret and frame is not None:
                        frame = cv2.resize(frame, (320, 240))
                        cv2.imshow("Camera View", frame)

                    # --- Non-blocking key input ---
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('v'):
                        self.vision_mode = True
                        print("Vision mode ON")
                    elif key == ord('q'):
                        print("Quit")
                        break
                    elif key == ord('w'):
                        self.manual_control('w')
                    elif key == ord('s'):
                        self.manual_control('s')
                    elif key == ord('a'):
                        self.manual_control('a')
                    elif key == ord('d'):
                        self.manual_control('d')
                    elif key == ord('i'):
                        self.manual_control('i')
                    elif key == ord('k'):
                        self.manual_control('k')
                    elif key == ord('j'):
                        self.manual_control('j')
                    elif key == ord('l'):
                        self.manual_control('l')
                    elif key == ord(' '):
                        self.manual_control(' ')
                    else:
                        self.px.forward(0)

        finally:
            self.shutdown()




if __name__ == "__main__":
    controller = PicarXController()
    controller.run()
import cv2
import numpy as np
from ultralytics import YOLO
from picarx import Picarx
from time import sleep
import threading
import time


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

        self.model = YOLO("/home/vehicle0/test_vehicle/best.pt")
        self.class_names = [
            '-Road narrows on right', '50 mph speed limit', 'Attention Please-', 'Beware of children',
            'CYCLE ROUTE AHEAD WARNING', 'Dangerous Left Curve Ahead', 'Dangerous Rright Curve Ahead',
            'End of all speed and passing limits', 'Give Way', 'Go Straight or Turn Right',
            'Go straight or turn left', 'Keep-Left', 'Keep-Right', 'Left Zig Zag Traffic',
            'No Entry', 'No_Over_Taking', 'Overtaking by trucks is prohibited', 'Pedestrian Crossing',
            'Round-About', 'Slippery Road Ahead', 'Speed Limit 20 KMPh', 'Speed Limit 30 KMPh',
            'Stop_Sign', 'Straight Ahead Only', 'Traffic_signal', 'Truck traffic is prohibited',
            'Turn left ahead', 'Turn right ahead', 'Uneven Road'
        ]

        self._run_yolo = False
        self._yolo_input = None
        self.latest_detections = []
        self._yolo_lock = threading.Lock()
        self._yolo_thread = threading.Thread(target=self._yolo_worker, daemon=True)
        self._yolo_thread.start()

    def _yolo_worker(self):
        while True:
            if self._run_yolo and self._yolo_input is not None:
                roi_copy = self._yolo_input.copy()
                small_roi = cv2.resize(roi_copy, (160, 120))
                results = self.model(small_roi, verbose=False)
                detections = []

                scale_x = roi_copy.shape[1] / 160
                scale_y = roi_copy.shape[0] / 120

                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = (xyxy * [scale_x, scale_y, scale_x, scale_y]).astype(int)
                    detections.append({
                        "class_id": cls_id,
                        "confidence": conf,
                        "bbox": (x1, y1, x2, y2)
                    })

                with self._yolo_lock:
                    self.latest_detections = detections
                self._run_yolo = False

    def get_rois(self, frame):
        height, width = frame.shape[:2]
        traffic_roi = frame[0:int(0.4 * height), :]
        lane_roi = frame[int(0.4 * height):, :]
        return traffic_roi, lane_roi

    def binary_threshold(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        return binary

    def find_lane_lines(self, binary_warped, nwindows=9, margin=25, minpix=30):
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        midpoint = histogram.shape[0] // 2
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        window_height = binary_warped.shape[0] // nwindows

        leftx_current = leftx_base
        rightx_current = rightx_base
        left_lane_inds, right_lane_inds = [], []

        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        try:
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
        except:
            return None, None, None

        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        return left_fitx, right_fitx, ploty

    def draw_lane_overlay(self, original, binary_warped, left_fitx, right_fitx, ploty):
        lane_img = np.zeros_like(original)
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right)).astype(np.int32)
        cv2.fillPoly(lane_img, [pts], (0, 255, 0))
        result = cv2.addWeighted(original, 1, lane_img, 0.3, 0)
        return result

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            print("Failed to grab frame")
            return None, None, None, None

        frame = cv2.resize(frame, (320, 240))
        traffic_roi, lane_roi = self.get_rois(frame)

        h, w = lane_roi.shape[:2]
        roi_w = int(0.4 * w)
        x_center = w // 2
        x1 = x_center - roi_w // 2
        x2 = x_center + roi_w // 2
        lane_roi_cropped = lane_roi[:, x1:x2]

        binary = self.binary_threshold(lane_roi_cropped)
        left_fitx, right_fitx, ploty = self.find_lane_lines(binary)

        if left_fitx is not None and right_fitx is not None:
            road_center = (left_fitx[-1] + right_fitx[-1]) // 2
            mid_x = lane_roi_cropped.shape[1] // 2
            error = road_center - mid_x
            steering_value = error / (lane_roi_cropped.shape[1] // 2)
            steering_value = max(-1.0, min(1.0, steering_value))
            angle_range = self.maxAngle - self.minAngle
            angle = (steering_value + 1) / 2 * angle_range + self.minAngle
            lane_output = self.draw_lane_overlay(lane_roi_cropped, binary, left_fitx, right_fitx, ploty)
            cv2.circle(lane_output, (int(road_center), h - 20), 5, (0, 255, 0), -1)
            return angle, binary, lane_output, traffic_roi

        return None, binary, lane_roi_cropped, traffic_roi

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()


class PicarXController:
    def __init__(self):
        self.px = Picarx()
        self.pan_angle = 0
        self.tilt_angle = 0
        self.speed = 0.1
        self.vision_mode = False
        self.vision = AckermannVision()
        self.frame_count = 0

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
        try:
            angle, mask, lane_output, traffic_roi = self.vision.process_frame()
        except Exception as e:
            print(f"[VisionControl Error] {e}")
            self.vision_mode = False
            return

        if lane_output is None:
            print("No frame!")
            return

        if angle is not None:
            steering_angle = int(np.clip(angle, -30, 30))
            print(f"[Vision] Steering Angle: {steering_angle}°")
            self.px.set_dir_servo_angle(steering_angle)
            self.px.forward(self.speed)
        else:
            print("Lane lost!")
            self.px.set_dir_servo_angle(0)
            self.px.forward(0)

        self.frame_count += 1
        if self.frame_count % 10 == 0 and traffic_roi is not None and not self.vision._run_yolo:
            self.vision._yolo_input = traffic_roi.copy()
            self.vision._run_yolo = True

        with self.vision._yolo_lock:
            detections = list(self.vision.latest_detections)

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cls_id = det["class_id"]
            conf = det["confidence"]
            label = f"{self.vision.class_names[cls_id]} ({conf:.2f})"
            cv2.rectangle(traffic_roi, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(traffic_roi, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            print(f"[YOLO DETECT] {label}")
            if self.vision.class_names[cls_id] == "Stop_Sign" and conf > 0.8:
                print("STOP SIGN DETECTED — STOPPING!")
                self.px.forward(0)

        cv2.imshow("Lane Detection", lane_output)
        if traffic_roi is not None:
            cv2.imshow("Traffic ROI", traffic_roi)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.shutdown()
        elif key == ord('v'):
            self.vision_mode = False
        elif key != 255:
            self.manual_control(chr(key))

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

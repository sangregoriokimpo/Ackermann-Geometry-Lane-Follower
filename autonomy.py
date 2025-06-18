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

        self.minAngle = 35
        self.maxAngle = 110
        self.wheelbase = 0.1  # meters
        self.pixels_per_meter = 300  # calibration factor

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            print("Failed to grab frame")
            return None, None, None, None

        frame = cv2.resize(frame, (320, 240))
        roi = frame[160:240, :]

        # Black line detection
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Contour detection for centerline
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lane_width_px = None

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                error = cx - (roi.shape[1] // 2)
                steering_value = error / (roi.shape[1] // 2)
                steering_value = max(-1.0, min(1.0, steering_value))
                angle_range = self.maxAngle - self.minAngle
                angle = (steering_value + 1) / 2 * angle_range + self.minAngle

                cv2.drawContours(roi, [largest_contour], -1, (0, 0, 255), 2)
                cv2.circle(roi, (cx, roi.shape[0] // 2), 5, (0, 255, 0), -1)

                x, y, w, h = cv2.boundingRect(largest_contour)
                lane_width_px = w

                return angle, mask, roi, lane_width_px

        return None, mask, roi, lane_width_px

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()


# -------- PiCar-X Controller ---------
class PicarXController:
    PAN_LIMIT = 35
    TILT_LIMIT = 35

    def __init__(self):
        self.px = Picarx()
        self.pan_angle = 0
        self.tilt_angle = 0
        self.speed = 80
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
            self.px.set_dir_servo_angle(-35)
            self.px.forward(self.speed)
        elif key == 'd':
            self.px.set_dir_servo_angle(35)
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

        self.px.set_cam_tilt_angle(self.tilt_angle)
        self.px.set_cam_pan_angle(self.pan_angle)

    def vision_control(self):
        angle, mask, roi, lane_width_px = self.vision.process_frame()
        if roi is None:
            print("No frame!")
            return

        if angle is not None:
            print(f"Steering Angle: {angle:.2f}")
            steering_angle = int(angle) - 75  # Centering
            steering_angle = np.clip(steering_angle, -35, 35)
            self.px.set_dir_servo_angle(steering_angle)
            self.px.forward(self.speed)
        else:
            print("Lane lost!")
            self.px.forward(0)

        cv2.imshow("Vision", roi)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.shutdown()

    def shutdown(self):
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
            ctrl+c: quit
        """)
        try:
            while True:
                if self.vision_mode:
                    self.vision_control()
                else:
                    key = readchar.readkey().lower()
                    if key == 'v':
                        self.vision_mode = not self.vision_mode
                        print(f"Vision mode {'ON' if self.vision_mode else 'OFF'}")
                        self.px.forward(0)
                    elif key == readchar.key.CTRL_C:
                        print("Quit")
                        break
                    else:
                        self.manual_control(key)
                        sleep(0.2)
                        self.px.forward(0)

        finally:
            self.shutdown()


if __name__ == "__main__":
    controller = PicarXController()
    controller.run()

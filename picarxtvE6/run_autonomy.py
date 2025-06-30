import cv2
import numpy as np
import math
from time import sleep
from picarx import Picarx
from ackermannVision import AckermannVision  # Local module
from kalmanFilter import KalmanFilter

class PicarXController:
    PAN_LIMIT = 35
    TILT_LIMIT = 35

    def __init__(self):
        print("Initializing PicarX controller...")
        self.px = Picarx()

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.kalman = KalmanFilter()

        if not self.cap.isOpened():
            raise RuntimeError("Camera failed to open. Check if it’s already in use.")

        self.pan_angle = 0
        self.tilt_angle = 0
        self.speed = 0.01
        self.vision_mode = False
        self.vision = AckermannVision(self.cap)

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
        try:
            angle, white_mask, overlay, black_mask = self.vision.process_frame()

            if overlay is None:
                print("No frame!")
                self.px.forward(0)
                return

            if angle is not None:
                filtered_angle = self.kalman.update(angle)
                steering_angle = int(np.clip(filtered_angle, -30, 30))

                print(f"[Vision] Steering Angle: {steering_angle}°")
                self.px.set_dir_servo_angle(steering_angle)
                self.px.forward(self.speed)
            else:
                print("Lane lost! Stopping.")
                self.px.forward(0)
                self.px.set_dir_servo_angle(0)

            cv2.imshow("Vision", overlay)
            cv2.imshow("White Mask", white_mask)
            cv2.imshow("Black Lane Mask", black_mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Exiting vision mode.")
                self.vision_mode = False

        except Exception as e:
            print("[ERROR] Exception in vision_control():", e)
            self.px.forward(0)




    def shutdown(self):
        print("Shutting down...")
        self.px.set_cam_tilt_angle(0)
        self.px.set_cam_pan_angle(0)
        self.px.set_dir_servo_angle(0)
        self.px.stop()
        self.cap.release()
        cv2.destroyAllWindows()
        sleep(0.2)
        exit()

    def run(self):
        print("""
        Control keys:
            w/s: forward/backward
            a/d: left/right turn
            i/k: camera tilt up/down
            j/l: camera pan left/right
            space: stop
            v: toggle vision mode
            q (in vision mode): quit
        """)
        try:
            while True:
                if self.vision_mode:
                    self.vision_control()
                else:
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        frame = cv2.resize(frame, (320, 240))
                        cv2.imshow("Camera View", frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('v'):
                        self.vision_mode = True
                        print("Vision mode ON")
                    elif key == ord('q'):
                        print("Quit")
                        break
                    elif key in [ord(c) for c in 'wasdijkl ']:
                        self.manual_control(chr(key))
                    else:
                        self.px.forward(0)
        finally:
            self.shutdown()

if __name__ == "__main__":
    controller = PicarXController()
    controller.run()
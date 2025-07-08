from picarx import Picarx
from vision import AckermannVision
from kalmanFilter import KalmanFilter
from trafficVision import TrafficVision
import cv2
from time import sleep, time

class PicarXController:
    def __init__(self):
        self.px = Picarx()
        self.vision = AckermannVision()
        self.kalman = KalmanFilter()
        self.traffic = TrafficVision()
        self.px.set_dir_servo_angle(0)
        self.mode = "manual"
        self.speed = 0.1

    def shutdown(self):
        print("Shutting down...")
        self.px.stop()
        self.vision.release()
        cv2.destroyAllWindows()
        sleep(0.2)
        exit()

    def run(self):
        print("""
        Modes:
        w/a/d/s - Manual drive
        space   - Stop
        v       - Vision lane follow
        q       - Quit
        """)

        try:
            while True:
                angle, lane_img, sign_frame = self.vision.process_frame()

                if angle is None and lane_img is None and sign_frame is None:
                    continue  # Camera frame failed; skip this loop

                if lane_img is not None:
                    cv2.imshow("Lane Detection", lane_img)
                if sign_frame is not None:
                    cv2.imshow("Sign Frame", sign_frame)

                if angle is not None and self.mode == "vision":
                    smoothed_angle = self.kalman.update(angle)
                    self.px.set_dir_servo_angle(int(smoothed_angle))
                    self.px.forward(self.speed)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.shutdown()
                elif key == ord(' '):
                    self.px.forward(0)
                    self.mode = "manual"
                elif key == ord('w'):
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
                    self.mode = "vision"
                    self.kalman = KalmanFilter()
                    print("Mode: VISION")

        finally:
            self.shutdown()

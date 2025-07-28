#picarxcontroller.py

from picarx import Picarx
from vision import AckermannVision
from kalmanFilter import KalmanFilter
from trafficVision import TrafficVision
from behaviours.stopSignHandler import StopSignHandler
from behaviours.speedLimitSignHandler import SpeedLimitSignHandler
from robot_hat import Music,TTS

import cv2
from time import sleep, time

class PicarXController:
    def __init__(self):
        self.px = Picarx()
        self.vision = AckermannVision()
        self.kalman = KalmanFilter()
        self.traffic = TrafficVision()
        self.stopHandler = StopSignHandler(cooldown = 10)
        self.speedHandler = SpeedLimitSignHandler()
        self.music = Music()
        self.music.music_set_volume(100)
        # tts.lang("en-US")


        self.px.set_dir_servo_angle(0)
        self.mode = "manual"
        self.speed = 0.1

        # Stop sign handling
        self.stop_sign_recently_handled = False
        self.last_stop_time = 0
        self.stop_sign_cooldown = 10  # seconds

        # --- Handle speed limits ---
        # new_speed = self.speedHandler.update_speed(detected_labels)
        # self.speed = new_speed


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
        v       - Vision lane follow (with Kalman smoothing)
        m       - Manual operation
        q       - Quit
        """)
        frame_count = 0
        process_every_n = 5  # Adjust to reduce detection frequency

        try:
            while True:
                frame_count += 1
                angle, lane_img, sign_frame = self.vision.process_frame()

                if sign_frame is None or sign_frame.size == 0:
                    continue

                current_time = time()

                # Run YOLO detection every N frames
                if frame_count % process_every_n == 0:
                    detection_img, detected_labels = self.traffic.draw_detections(sign_frame)
                    cv2.imshow("Traffic Sign Detection", detection_img)
                
                    if self.stopHandler.should_stop(detected_labels):
                        self.stopHandler.handle_stop(self.px)

                    # --- Handle speed limits ---
                    new_speed = self.speedHandler.update_speed(detected_labels)
                    self.speed = new_speed

                # Lane view
                if lane_img is not None and lane_img.size > 0:
                    cv2.imshow("Lane Detection", lane_img)

                # Safe Kalman filtering
                if angle is not None:
                    smoothed_angle = self.kalman.update(angle)
                    if self.mode == "vision":
                        self.px.set_dir_servo_angle(int(smoothed_angle))
                        self.px.forward(self.speed)
                    print(f"ANGLE: {angle:.2f}, KF: {smoothed_angle:.2f}")
                else:
                    print("ANGLE: None — skipping Kalman update")


                # User controls
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.shutdown()
                elif key == ord(' '):
                    self.px.forward(0)
                    self.px.set_dir_servo_angle(0)
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
                    print("[Mode]: VISION")
                    self.music.sound_play("voicelines/visionA1.mp3")
                elif key == ord('m'):
                    self.px.forward(0)
                    self.mode = "manual"
                    print("[Mode]: MANUAL")
                    self.music.sound_play("voicelines/manualA1.mp3")

                # elif key == ord('v'):
                #     if self.mode == "vision":
                #         self.px.forward(0)
                #         self.mode = "manual"
                #         print("Mode: MANUAL")
                #     else:
                #         self.kalman = KalmanFilter()
                #         self.mode = "vision"
                #         print("Mode: VISION")


        finally:
            self.shutdown()
import cv2
import os
import datetime
from picarx import Picarx
from time import sleep

class PicarXRecorder:
    def __init__(self, device='/dev/video0', output_dir=None):
        # Set output directory relative to the script file
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), 'recordings')
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize PiCar-X
        self.px = Picarx()
        self.speed = 0.1

        # Initialize camera
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FPS, 5)

        if not self.cap.isOpened():
            raise IOError(f"Cannot open camera {device}")

        # Recording state
        self.recording = False
        self.writer = None

    def toggle_recording(self, frame_size=(320, 240)):
        if not self.recording:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f"recording_{timestamp}.mp4")
            print(f"[REC] Recording to: {os.path.abspath(filename)}")
            self.writer = cv2.VideoWriter(
                filename,
                cv2.VideoWriter_fourcc(*'MJPG'),
                30.0,
                frame_size
            )
            if not self.writer.isOpened():
                print("[ERROR] Failed to open VideoWriter!")
                self.writer = None
            else:
                self.recording = True
                print(f"[REC] Started recording: {filename}")
        else:
            print("[REC] Stopping recording")
            self.recording = False
            if self.writer:
                self.writer.release()
                self.writer = None
            print("[REC] Stopped recording")

    def manual_control(self, key):
        if key == ord('w'):
            print("Move forward")
            self.px.set_dir_servo_angle(0)
            self.px.forward(self.speed)
        elif key == ord('s'):
            print("Move backward")
            self.px.set_dir_servo_angle(0)
            self.px.backward(self.speed)
        elif key == ord('a'):
            print("Turn left")
            self.px.set_dir_servo_angle(-30)
            self.px.forward(self.speed)
        elif key == ord('d'):
            print("Turn right")
            self.px.set_dir_servo_angle(30)
            self.px.forward(self.speed)
        elif key == ord(' '):
            print("Stopping car")
            self.px.forward(0)
            self.px.set_dir_servo_angle(0)
            self.toggle_recording()
        else:
            self.px.forward(0)

    def run(self):
        print("""
        PiCar-X WASD Controller + Video Recorder

        Controls:
            w - Forward
            s - Backward
            a - Left
            d - Right
            space - Stop + Toggle Recording
            q - Quit
        """)
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    print("[ERROR] Failed to grab frame")
                    continue

                # Display camera feed
                cv2.imshow("Live Feed", frame)

                # Save frame if recording
                if self.recording and self.writer:
                    self.writer.write(frame)

                # Get key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Exiting...")
                    break
                elif key != 255:  # If a key was pressed
                    self.manual_control(key)

        finally:
            self.shutdown()

    def shutdown(self):
        print("Shutting down PiCar-X controller...")
        self.px.forward(0)
        self.px.set_dir_servo_angle(0)
        sleep(0.1)
        if self.writer:
            self.writer.release()
        self.cap.release()
        cv2.destroyAllWindows()
        print("All resources released.")


if __name__ == "__main__":
    controller = PicarXRecorder()
    controller.run()

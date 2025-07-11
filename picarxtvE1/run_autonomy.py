import cv2
import time
import os
import sys
import subprocess
from picarxtvE1.vision import LineFollower
from picarx import Picarx

# Start pigpiod safely
def start_pigpiod():
    try:
        subprocess.run(["pgrep", "pigpiod"], check=True)
        print("pigpiod is already running.")
    except subprocess.CalledProcessError:
        print("🔧 Starting pigpiod...")
        subprocess.run(["sudo", "pigpiod"])
        time.sleep(1)

def main():
    start_pigpiod()

    # Initialize PiCar-X
    px = Picarx()

    # Start camera
    cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    if not cap.isOpened():
        print("Failed to open camera.")
        sys.exit(1)

    follower = LineFollower()

    max_angle = 30
    drive_speed = 60  # PWM 0–100

    print("Autonomous driving started. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            angle, mask, roi = follower.process_frame(frame)

            if angle is not None:
                angle = max(-max_angle, min(max_angle, angle))
                px.set_dir_servo_angle(angle)
                px.forward(drive_speed)
                print(f"Steering to {angle:.2f}°, speed {drive_speed}")
            else:
                px.stop()
                print("Line lost! Car stopped.")

            cv2.imshow("Mask", mask)
            cv2.imshow("ROI", roi)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        cap.release()
        px.stop()
        px.set_dir_servo_angle(0)
        cv2.destroyAllWindows()
        print("Car shutdown complete.")

if __name__ == "__main__":
    main()

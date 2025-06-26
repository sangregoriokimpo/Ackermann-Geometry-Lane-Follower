import subprocess
import os
import sys
import cv2
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONTROLLER_PATH = os.path.join(BASE_DIR, "controller.py")

# Use an Event to safely signal the camera thread to stop
stop_event = threading.Event()

def camera_feed():
    print("Starting camera...")
    cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)

    if not cap.isOpened():
        print("Failed to open camera.")
        return

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        cv2.imshow("PiCar-X Camera Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera feed stopped.")

def start_teleop():
    print("Starting teleoperation...")

    cam_thread = threading.Thread(target=camera_feed, daemon=True)
    cam_thread.start()

    try:
        subprocess.run(["python3", CONTROLLER_PATH])
    except KeyboardInterrupt:
        print("\nController interrupted.")
    finally:
        stop_event.set()
        cam_thread.join()
        print("Teleoperation finished.")

if __name__ == "__main__":
    try:
        start_teleop()
    except KeyboardInterrupt:
        print("\nStopping teleop...")
        sys.exit(0)

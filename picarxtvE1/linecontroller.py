from picarxtlE1.hardware import Car
from picarxtlE1.vision import LineFollower
import cv2

car = Car()
follower = LineFollower()

cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

if not cap.isOpened():
    print("Failed to open camera.")
    exit(1)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        angle, mask, roi = follower.process_frame(frame)

        if angle is not None:
            car.move_steering(angle)
            car.drive(0.6)
            print(f"Angle: {angle:.2f}")
        else:
            car.stop()
            print("Line lost!")

        cv2.imshow("Mask", mask)
        cv2.imshow("ROI", roi)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    cap.release()
    car.stop()
    cv2.destroyAllWindows()

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
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
        rows, cols = image.shape[:2]
        vertices = np.array([[
            [cols * 0.1, rows * 0.95],
            [cols * 0.4, rows * 0.6],
            [cols * 0.6, rows * 0.6],
            [cols * 0.9, rows * 0.95]
        ]], dtype=np.int32)
        cv2.fillPoly(mask, vertices, 255)
        return cv2.bitwise_and(image, mask)

    def hough_transform(self, image):
        return cv2.HoughLinesP(image, 1, np.pi / 180, 20, minLineLength=20, maxLineGap=500)

    def average_slope_intercept(self, lines):
        left_lines, right_lines, left_weights, right_weights = [], [], [], []
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

    def pixel_points(self, y1, y2, line):
        if line is None:
            return None
        slope, intercept = line
        if abs(slope) < 0.01:  # avoid nearly horizontal lines
            return None
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return ((x1, int(y1)), (x2, int(y2)))

    def lane_lines(self, image, lines):
        left_lane, right_lane = self.average_slope_intercept(lines)
        y1 = image.shape[0]
        y2 = y1 * 0.6
        return self.pixel_points(y1, y2, left_lane), self.pixel_points(y1, y2, right_lane)

    def draw_lane_lines(self, image, lines, road_center=None):
        output = image.copy()
        for line in lines:
            if line is not None:
                cv2.line(output, *line, (255, 0, 0), 6)
        if road_center is not None:
            cv2.circle(output, (road_center, image.shape[0] - 20), 5, (0, 255, 0), -1)
        return output

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None, None, None

        frame = cv2.resize(frame, (320, 240))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        masked = self.region_selection(edges)
        lines = self.hough_transform(masked)

        angle = None
        road_center = None
        if lines is not None:
            left_line, right_line = self.lane_lines(frame, lines)
            mid_x = frame.shape[1] // 2

            if left_line and right_line:
                road_center = (left_line[1][0] + right_line[1][0]) // 2
            elif left_line:
                road_center = left_line[1][0] + 100
            elif right_line:
                road_center = right_line[1][0] - 100

            if road_center is not None:
                error = road_center - mid_x
                steering_value = error / mid_x
                steering_value = np.clip(steering_value, -1.0, 1.0)
                angle = ((steering_value + 1) / 2) * (self.maxAngle - self.minAngle) + self.minAngle

            lane_img = self.draw_lane_lines(frame, [left_line, right_line], road_center)
        else:
            lane_img = frame.copy()

        return angle, lane_img, frame[int(240*0.4):] / 255.0  # return cropped input for model


# -------- Neural Driving Agent --------
class DrivingAgent:
    def __init__(self):
        self.model = Sequential([
            Conv2D(32, (5,5), activation='relu', strides=(2,2), input_shape=(144, 320, 3)),
            MaxPooling2D(2,2),
            Conv2D(64, (3,3), activation='relu', strides=(2,2)),
            MaxPooling2D(2,2),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(1, activation='linear')
        ])
        self.model.compile(optimizer=Adam(1e-3), loss='mse')
        self.memory = []

    def predict(self, img):
        img = np.reshape(img, (1, 144, 320, 3))
        return self.model.predict(img, verbose=0)[0][0]

    def remember(self, img, angle):
        self.memory.append((img, angle / 30.0))  # Normalize

    def train(self, batch_size=8):
        if len(self.memory) < batch_size:
            return
        X, y = zip(*self.memory[-batch_size:])
        self.model.fit(np.array(X), np.array(y), epochs=1, verbose=0)
        print("[Train] Sample:", len(self.memory), "Loss:", self.model.evaluate(np.array(X), np.array(y), verbose=0))

    def save(self, path="lane_model.h5"):
        self.model.save(path)

    def load(self, path="lane_model.h5"):
        self.model = tf.keras.models.load_model(path)


# -------- PiCarX Controller --------
class PicarXController:
    def __init__(self):
        self.px = Picarx()
        self.vision = AckermannVision()
        self.agent = DrivingAgent()
        self.mode = "manual"  # options: manual, vision, train, model
        self.speed = 0.1

    def shutdown(self):
        print("Shutting down...")
        self.px.stop()
        self.vision.release()
        self.agent.save()
        sleep(0.2)
        exit()

    def run(self):
        print("""
Modes:
  w/a/d/s - Manual drive
  space   - Stop
  v       - Vision lane follow (based on green dot)
  t       - Train model from vision steering
  m       - Drive using model (autonomous)
  q       - Quit
""")
        try:
            while True:
                angle, lane_img, img_input = self.vision.process_frame()
                if lane_img is not None:
                    cv2.imshow("Lane View", lane_img)

                if self.mode == "vision" and angle is not None:
                    self.px.set_dir_servo_angle(int(angle))
                    self.px.forward(self.speed)

                elif self.mode == "train" and angle is not None:
                    self.agent.remember(img_input, angle)
                    self.agent.train()
                    self.px.set_dir_servo_angle(int(angle))
                    self.px.forward(self.speed)

                elif self.mode == "model" and img_input is not None:
                    pred = self.agent.predict(img_input)
                    pred_angle = np.clip(pred, -1.0, 1.0) * 30
                    print(f"[Model] Steering Angle: {pred_angle:.2f}")
                    self.px.set_dir_servo_angle(int(pred_angle))
                    self.px.forward(self.speed)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.shutdown()
                elif key == ord(' '):
                    self.px.forward(0)
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
                    self.mode = "vision"
                    print("Mode: VISION")
                elif key == ord('t'):
                    self.mode = "train"
                    print("Mode: TRAIN (CNN learning from green dot)")
                elif key == ord('m'):
                    self.mode = "model"
                    print("Mode: MODEL (Autonomous CNN driving)")
        finally:
            self.shutdown()


if __name__ == "__main__":
    controller = PicarXController()
    controller.run()
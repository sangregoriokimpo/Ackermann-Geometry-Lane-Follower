import cv2
import numpy as np
from picarx import Picarx
from time import sleep


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
        ignore_mask_color = 255
        rows, cols = image.shape[:2]

        # Use bottom 60% of image as ROI
        top_y = int(rows * 0.4)
        vertices = np.array([[
            (0, rows),
            (0, top_y),
            (cols, top_y),
            (cols, rows)
        ]], dtype=np.int32)

        cv2.fillPoly(mask, vertices, ignore_mask_color)
        return cv2.bitwise_and(image, mask)

    def pipeline_threshold(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv, (15, 100, 100), (35, 255, 255))
        white_mask = cv2.inRange(image, (200, 200, 200), (255, 255, 255))
        combined = cv2.bitwise_or(yellow_mask, white_mask)
        blurred = cv2.GaussianBlur(combined, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        masked = self.region_selection(edges)
        return masked

    def sliding_window_polyfit(self, binary_warped):
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
        midpoint = histogram.shape[0] // 2
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        nwindows = 9
        window_height = binary_warped.shape[0] // nwindows
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        leftx_current = leftx_base
        rightx_current = rightx_base
        margin = 25
        minpix = 30
        left_lane_inds = []
        right_lane_inds = []

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

        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

        try:
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            center_offset = ((left_fitx[-1] + right_fitx[-1]) / 2) - binary_warped.shape[1] / 2
            return binary_warped, left_fitx, right_fitx, ploty, center_offset
        except:
            return binary_warped, None, None, ploty, None

    def draw_curved_lanes(self, image, left_fitx, right_fitx, ploty):
        lane_img = np.zeros_like(image)
        if left_fitx is None or right_fitx is None:
            return image

        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right)).astype(np.int32)

        cv2.fillPoly(lane_img, [pts], (0, 255, 0))
        combined = cv2.addWeighted(image, 1, lane_img, 0.3, 0)
        return combined

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret or frame is None:
            print("Failed to grab frame")
            return None, None, None, None

        frame = cv2.resize(frame, (320, 240))
        binary_warped = self.pipeline_threshold(frame)
        out_img, left_fitx, right_fitx, ploty, center_offset = self.sliding_window_polyfit(binary_warped)

        if center_offset is not None:
            steering_value = center_offset / (frame.shape[1] // 2)
            steering_value = max(-1.0, min(1.0, steering_value))
            angle_range = self.maxAngle - self.minAngle
            angle = (steering_value + 1) / 2 * angle_range + self.minAngle
        else:
            angle = None

        lane_image = self.draw_curved_lanes(frame, left_fitx, right_fitx, ploty)
        return angle, binary_warped, lane_image, None

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()


# -------- PiCar-X Controller ---------
class PicarXController:
    PAN_LIMIT = 35
    TILT_LIMIT = 35

    def __init__(self):
        print("Initializing PicarX controller...")
        self.px = Picarx()
        self.pan_angle = 0
        self.tilt_angle = 0
        self.speed = 0.1
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
        angle, mask, frame_out, _ = self.vision.process_frame()
        if frame_out is None:
            print("No frame!")
            return

        if angle is not None:
            steering_angle = int(np.clip(angle, -30, 30))
            print(f"[Vision] Steering Angle: {steering_angle}°")
            self.px.set_dir_servo_angle(steering_angle)
            self.px.forward(self.speed)
        else:
            print("Lane lost! Stopping car.")
            self.px.forward(0)
            self.px.set_dir_servo_angle(0)

        cv2.imshow("Vision", frame_out)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.shutdown()

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
            w: forward
            s: backward
            a/d: left/right turn
            i/k: head tilt up/down
            j/l: head pan left/right
            v: toggle vision mode
            space: stop
            ctrl+c or q (in vision): quit
        """)
        try:
            while True:
                if self.vision_mode:
                    self.vision_control()
                else:
                    ret, frame = self.vision.cap.read()
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
                    elif key == ord('w'):
                        self.manual_control('w')
                    elif key == ord('s'):
                        self.manual_control('s')
                    elif key == ord('a'):
                        self.manual_control('a')
                    elif key == ord('d'):
                        self.manual_control('d')
                    elif key == ord('i'):
                        self.manual_control('i')
                    elif key == ord('k'):
                        self.manual_control('k')
                    elif key == ord('j'):
                        self.manual_control('j')
                    elif key == ord('l'):
                        self.manual_control('l')
                    elif key == ord(' '):
                        self.manual_control(' ')
                    else:
                        self.px.forward(0)
        finally:
            self.shutdown()


if __name__ == "__main__":
    controller = PicarXController()
    controller.run()

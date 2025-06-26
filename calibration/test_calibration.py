from picarx import Picarx
import time
px = Picarx()
px.set_dir_servo_angle(0)
time.sleep(3)
px.set_dir_servo_angle(-30)
time.sleep(3)
px.set_dir_servo_angle(30)
time.sleep(3)
px.set_dir_servo_angle(0)

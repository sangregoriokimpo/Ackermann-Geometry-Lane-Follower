from picarx import Picarx
import readchar
import time

# Initialize PiCar-X
px = Picarx()

# Automatically read stored steering calibration offset
STEERING_CALIBRATION = px.dir_cali_val

# Control parameters
steering_angle = 0
drive_power = 0
steering_step = 5
drive_step = 10
max_drive_power = 100
max_steering_angle = 30

print("""
PiCar-X Manual Control:
  w - increase speed
  s - decrease speed
  a - steer left
  d - steer right
  space - reset steering to center
  q - quit
""")
print("Steering Offset: ,{STEERING_CALIBRATION}")

# Apply steering with calibration compensation
# def set_steering(angle):
#     # Compensate for calibration offset
#     px.set_dir_servo_angle(angle - STEERING_CALIBRATION)

try:
    # Ensure wheels are centered at start
    px.set_dir_servo_angle(0)


    while True:
        key = readchar.readkey().lower()

        if key == 'a':
            steering_angle -= steering_step
            steering_angle = max(-max_steering_angle, steering_angle)
            px.set_dir_servo_angle(steering_angle)

        elif key == 'd':
            steering_angle += steering_step
            steering_angle = min(max_steering_angle, steering_angle)
            px.set_dir_servo_angle(steering_angle)


        elif key == ' ':
            steering_angle = 0
            px.set_dir_servo_angle(steering_angle)


        elif key == 'w':
            drive_power += drive_step
            drive_power = min(max_drive_power, drive_power)

        elif key == 's':
            drive_power -= drive_step
            drive_power = max(-max_drive_power, drive_power)

        elif key == 'q':
            break

        # Apply driving power
        if drive_power > 0:
            px.forward(drive_power)
        elif drive_power < 0:
            px.backward(abs(drive_power))
        else:
            px.stop()

        time.sleep(0.05)

except KeyboardInterrupt:
    pass

finally:
    px.stop()
    px.set_dir_servo_angle(0)

    print("Program exited cleanly.")

import sys
import tty
import termios
import select
import time
from hardware import Car

car = Car()
step = 5  # degrees per key press
power_step = 0.1  # drive power change

def is_data():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

def get_key():
    if is_data():
        return sys.stdin.read(1)
    return None

print("WASD to drive, SPACE to reset steering, Q to quit")

fd = sys.stdin.fileno()
old_settings = termios.tcgetattr(fd)
tty.setcbreak(fd)

try:
    while True:
        key = get_key()
        if key:
            if key == 'a':
                car.move_steering(car.current_angle - step)
            elif key == 'd':
                car.move_steering(car.current_angle + step)
            elif key == 'w':
                car.drive(power_step)
            elif key == 's':
                car.drive(-power_step)
            elif key == ' ':
                car.reset_steering()
            elif key == 'q':
                break
            else:
                print("Invalid key.")
        time.sleep(0.05)

except KeyboardInterrupt:
    pass

finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    car.stop()

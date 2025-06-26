from picarx import Picarx

class Car:
    def __init__(self):
        self.px = Picarx()
        self.minAngle = -30
        self.maxAngle = 30
        self.current_angle = 0
        self.drive_power = 0
        self.move_steering(0)

    def move_steering(self, angle):
        angle = max(self.minAngle, min(self.maxAngle, angle))
        self.current_angle = angle
        self.px.set_dir_servo_angle(angle)
        print(f"Steering moved to {self.current_angle:.2f}°")

    def reset_steering(self):
        self.move_steering(0)

    def drive(self, power):
        self.drive_power = max(-1.0, min(1.0, power))
        if self.drive_power > 0:
            self.px.forward(self.drive_power * 100)
        elif self.drive_power < 0:
            self.px.backward(abs(self.drive_power) * 100)
        else:
            self.px.stop()
        print(f"Drive power: {self.drive_power:.2f}")

    def stop(self):
        self.px.stop()
        self.reset_steering()
        print("Car stopped.")

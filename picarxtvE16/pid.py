#pid.py

class PIDController:
    def __init__(self, Kp=0.5, Ki=0.01, Kd=0.1, integral_limit=100.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0.0
        self.prev_error = 0.0
        self.integral_limit = integral_limit

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error, dt):
        # Integrate with clamping
        self.integral += error * dt
        self.integral = max(min(self.integral, self.integral_limit), -self.integral_limit)

        # Derivative
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0

        # PID output
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        self.prev_error = error
        return output

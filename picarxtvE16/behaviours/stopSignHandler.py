# behaviors/stopSignHandler.py

from time import sleep, time

class StopSignHandler:
    def __init__(self, cooldown=10):
        self.last_stop_time = 0
        self.cooldown = cooldown
        self.recently_stopped = False

    def should_stop(self, labels_detected):
        return "Stop_Sign" in labels_detected

    def handle_stop(self, robot_drive):
        current_time = time()

        if not self.recently_stopped:
            print("🛑 Stop sign detected! Pausing for 3 seconds...")
            robot_drive.forward(0)
            sleep(3)
            self.last_stop_time = current_time
            self.recently_stopped = True

        elif current_time - self.last_stop_time > self.cooldown:
            # Cooldown expired
            self.recently_stopped = False

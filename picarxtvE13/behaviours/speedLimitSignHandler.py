# behaviours/speedLimitSignHandler.py
# 0: '20 mph sign',
# 1: '30 mph sign',
# 2: '50 mph sign',
# 3: 'stop sign'

class SpeedLimitSignHandler:
    def __init__(self):
        # Default speed if no sign seen
        self.current_speed = 0.1

        # Mapping of sign labels to speeds
        self.speed_map = {
            "20 mph sign": 2,
            "30 mph sign": 3,
            "50 mph sign": 5,
        }

    def update_speed(self, labels):
        for label in labels:
            if label in self.speed_map:
                self.current_speed = self.speed_map[label]
                print(f"Speed limit sign detected: {label} → Speed set to {self.current_speed}")
                break  # Only use first matching speed sign
        return self.current_speed

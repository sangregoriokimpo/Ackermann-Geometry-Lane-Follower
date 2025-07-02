# behaviours/speedLimitSignHandler.py

class SpeedLimitSignHandler:
    def __init__(self):
        # Default speed if no sign seen
        self.current_speed = 0.1

        # Mapping of sign labels to speeds
        self.speed_map = {
            "Speed Limit 20 KMPh": 0.5,
            "Speed Limit 30 KMPh": 1.0,
            "50 mph speed limit": 2.0,
        }

    def update_speed(self, labels):
        for label in labels:
            if label in self.speed_map:
                self.current_speed = self.speed_map[label]
                print(f"🚦 Speed limit sign detected: {label} → Speed set to {self.current_speed}")
                break  # Only use first matching speed sign
        return self.current_speed

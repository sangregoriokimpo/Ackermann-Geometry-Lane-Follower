# sign_detection.py
import threading
from ultralytics import YOLO

class SignDetector:
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        self.detection_result = None
        self.running = False
        self.thread = None

    def detect_async(self, frame):
        if self.thread and self.thread.is_alive():
            return  # Skip if detection still in progress
        self.thread = threading.Thread(target=self._detect, args=(frame,))
        self.thread.start()

    def _detect(self, frame):
        results = self.model(frame, verbose=False)[0]
        for cls_id in results.boxes.cls:
            cls = int(cls_id.item())
            if cls in [9, 11]:  # 9=traffic light, 11=stop sign
                self.detection_result = cls

    def get_result(self):
        return self.detection_result

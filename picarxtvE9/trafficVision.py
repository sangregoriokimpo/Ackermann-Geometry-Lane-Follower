# trafficVision.py

from ultralytics import YOLO
import cv2

class TrafficVision:
    def __init__(self):
        self.model = YOLO("models/best_640.pt")
        self.class_names = {
            0: '20 mph sign',
            1: '30 mph sign',
            2: '50 mph sign',
        }
        # self.class_names = {
        #     0: '-Road narrows on right',
        #     1: '50 mph speed limit',
        #     2: 'Attention Please-',
        #     3: 'Beware of children',
        #     4: 'CYCLE ROUTE AHEAD WARNING',
        #     5: 'Dangerous Left Curve Ahead',
        #     6: 'Dangerous Rright Curve Ahead',
        #     7: 'End of all speed and passing limits',
        #     8: 'Give Way',
        #     9: 'Go Straight or Turn Right',
        #     10: 'Go straight or turn left',
        #     11: 'Keep-Left',
        #     12: 'Keep-Right',
        #     13: 'Left Zig Zag Traffic',
        #     14: 'No Entry',
        #     15: 'No_Over_Taking',
        #     16: 'Overtaking by trucks is prohibited',
        #     17: 'Pedestrian Crossing',
        #     18: 'Round-About',
        #     19: 'Slippery Road Ahead',
        #     20: 'Speed Limit 20 KMPh',
        #     21: 'Speed Limit 30 KMPh',
        #     22: 'Stop_Sign',
        #     23: 'Straight Ahead Only',
        #     24: 'Traffic_signal',
        #     25: 'Truck traffic is prohibited',
        #     26: 'Turn left ahead',
        #     27: 'Turn right ahead',
        #     28: 'Uneven Road'
        # }

    def draw_detections(self, frame):
        results = self.model(frame, conf=0.1, verbose=False)[0]
        labels_detected = set()

        for box in results.boxes:
            cls_id = int(box.cls)
            label = self.class_names.get(cls_id, "Unknown")
            labels_detected.add(label)

            # Draw the box
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return frame, labels_detected

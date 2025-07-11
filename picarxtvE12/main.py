#main.py

from picarxcontroller import PicarXController
from trafficVision import TrafficVision

if __name__ == "__main__":
    controller = PicarXController()
    controller.run()

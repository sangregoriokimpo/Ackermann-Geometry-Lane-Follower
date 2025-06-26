import cv2
import numpy as np

img = np.zeros((240, 320, 3), dtype=np.uint8)
cv2.putText(img, "Test Display", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
cv2.imshow("Test", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

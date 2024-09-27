import cv2
from picamera2 import Picamera2

# Initialize Picamera2
picam2 = Picamera2()
picam2.start()

while True:
    frame = picam2.capture_array()
    cv2.imshow('Camera Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

from flask import Flask, render_template, Response
import cv2
from picamera2 import Picamera2
import numpy as np

# Load pre-trained model for AI detection (e.g., a MobileNet or a custom model)
# For example, using a pre-trained MobileNet from OpenCV
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet.caffemodel')

# Class labels for image recognition (specific to the model you're using)
# Ensure the class labels match the model you are using
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

app = Flask(__name__)

picam2 = Picamera2()
picam2.start()

def gen_frames():
    while True:
        # Capture the frame from the Pi camera
        frame = picam2.capture_array()

        # Ensure the frame has 3 channels (convert RGBA to RGB if necessary)
        if frame.shape[2] == 4:  # Check if there are 4 channels (RGBA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # Convert to RGB

        # Convert the image into a blob for object detection
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        net.setInput(blob)

        # Perform detection
        detections = net.forward()

        # Loop over the detections to display the results
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Confidence threshold
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array(
                    [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")

                # Draw the bounding box around the detected object and add the label
                label = f"{CLASSES[idx]}: {confidence:.2f}"
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Encode the frame back into a JPEG format for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame to the Flask web server
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return "Stream available at /video_feed"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

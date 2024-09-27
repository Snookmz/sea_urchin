from flask import Flask, render_template, Response
import cv2
from picamera2 import Picamera2

app = Flask(__name__)

picam2 = Picamera2()
picam2.start()

def gen_frames():
    while True:
        frame = picam2.capture_array()
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
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

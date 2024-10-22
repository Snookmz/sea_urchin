from flask import Flask, render_template, Response
import cv2
from picamera2 import Picamera2
import numpy as np
import tflite_runtime.interpreter as tflite

# Load TensorFlow Lite model
interpreter = tflite.Interpreter(model_path='models/real_sea_urchin.tflite')  # Replace with your TFLite model path
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize Flask app
app = Flask(__name__)

# Initialize Pi camera
picam2 = Picamera2()
camera_config = picam2.create_still_configuration(main={"size": (640, 480)})  # Lower resolution (640x480)
picam2.configure(camera_config)
picam2.start()

# Function to preprocess image for TensorFlow Lite model
def preprocess_image(frame, input_shape):
    # Resize the image to the input size of the model
    resized_frame = cv2.resize(frame, (input_shape[1], input_shape[2]))
    # Normalize and expand dimensions to match the model's input
    input_data = np.expand_dims(resized_frame, axis=0).astype(np.float32) / 255.0
    return input_data

def gen_frames():
    while True:
        # Capture the frame from the Pi camera
        frame = picam2.capture_array()

        # Ensure the frame is in RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Preprocess the frame for the TensorFlow Lite model
        input_data = preprocess_image(frame_rgb, input_details[0]['shape'])

        # Perform inference with TensorFlow Lite model
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # Get the prediction result
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        predicted_class = np.argmax(output_data)  # Class with the highest probability

        # Load class names (assuming 0 is 'sea urchin' and 1 is 'not sea urchin')
        class_names = ['Sea Urchin', 'Not Sea Urchin']
        label = class_names[predicted_class]

        # Display the label on the frame
        cv2.putText(frame, f'Prediction: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode the frame back into a JPEG format for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame to the Flask web server
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route to stream video feed
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Index route
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

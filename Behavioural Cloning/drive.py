import socketio
import eventlet
import numpy as np
from flask import Flask
from tensorflow.keras.models import load_model  # Import load_model directly
import base64
from io import BytesIO
from PIL import Image
import cv2

# Initialize Socket.IO server
sio = socketio.Server()

# Initialize Flask app
app = Flask(__name__)
speed_limit = 10

# Image preprocessing function
def img_preprocess(img):
    img = img[60:135, :, :]  # Crop the image
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)  # Convert to YUV
    img = cv2.GaussianBlur(img, (3, 3), 0)  # Apply Gaussian blur
    img = cv2.resize(img, (200, 66))  # Resize the image
    img = img / 255  # Normalize the image
    return img

# Socket.IO event for telemetry data
@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])  # Get the current speed
    image = Image.open(BytesIO(base64.b64decode(data['image'])))  # Decode the image
    image = np.asarray(image)  # Convert to NumPy array
    image = img_preprocess(image)  # Preprocess the image
    image = np.array([image])  # Add batch dimension
    steering_angle = float(model.predict(image))  # Predict steering angle
    throttle = 1.0 - speed / speed_limit  # Calculate throttle
    print(f'Steering Angle: {steering_angle}, Throttle: {throttle}, Speed: {speed}')
    send_control(steering_angle, throttle)  # Send control commands

# Socket.IO event for connection
@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    send_control(0, 0)  # Initialize with zero steering and throttle

# Function to send control commands
def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle': str(steering_angle),
        'throttle': str(throttle)
    })

# Load the model
model = load_model('model.h5')  # Load the model directly

# Start the server
if __name__ == '__main__':
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)


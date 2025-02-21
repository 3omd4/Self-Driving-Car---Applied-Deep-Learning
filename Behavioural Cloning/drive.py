import socketio
import eventlet
import tensorflow.keras as keras
import numpy as np
from flask import Flask
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2

# Initialize socket server
sio = socketio.Server()
app = Flask(__name__)

# Speed limit (adjust as needed)
speed_limit = 10

def img_preprocess(img):
    """Preprocess the image for the model."""
    img = img[60:135, :, :]  # Crop the image
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)  # Convert color space
    img = cv2.GaussianBlur(img, (3, 3), 0)  # Apply Gaussian blur
    img = cv2.resize(img, (200, 66))  # Resize to match model input
    img = img / 255.0  # Normalize
    return img

@sio.on('telemetry')
def telemetry(sid, data):
    """Handle telemetry data from the simulator."""
    if data is None:
        print("❌ No telemetry data received!")
        return

    try:
        print("✅ Telemetry Data Received")
        print(data.keys())  # Print available telemetry data keys

        speed = float(data['speed'])  # Get current speed
        image = Image.open(BytesIO(base64.b64decode(data['image'])))
        image = np.asarray(image)  # Convert image to numpy array
        image = img_preprocess(image)  # Preprocess image
        image = np.array([image])  # Expand dimensions for model input

        # Predict steering angle
        steering_angle = float(model.predict(image)[0][0])

        print(f"🔄 Predicted Steering Angle: {steering_angle:.4f}")

        # Calculate throttle (ensures car keeps moving)
        throttle = max(0.2, 1.0 - (speed / speed_limit))  
        print(f"🛞 Steering: {steering_angle:.4f}, 🚀 Throttle: {throttle:.2f}, ⚡ Speed: {speed}")

        send_control(steering_angle, throttle)

    except Exception as e:
        print(f"❌ Error in telemetry processing: {e}")

@sio.on('connect')
def connect(sid, environ):
    """Handle new simulator connection."""
    print('✅ Connected to simulator!')
    send_control(0.0, 0.5)  # Force slight movement on connect

def send_control(steering_angle, throttle):
    """Send control commands to the simulator."""
    print(f"📤 Sending Control - Steering: {steering_angle:.4f}, Throttle: {throttle:.2f}")
    sio.emit('steer', data={
        'steering_angle': str(steering_angle),
        'throttle': str(throttle)
    })

def mse(y_true, y_pred):
    """Custom Mean Squared Error loss function."""
    return keras.losses.mean_squared_error(y_true, y_pred)

if __name__ == '__main__':
    try:
        model = load_model('model.h5', custom_objects={'mse': mse})
        print("✅ Model loaded successfully!")
        model.summary()  # Print model architecture for verification
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        exit(1)

    # Start server
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

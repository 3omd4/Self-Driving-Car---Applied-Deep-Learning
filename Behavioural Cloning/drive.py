import socketio
import eventlet
import tensorflow.keras as keras
import base64
import numpy as np 
import cv2

from PIL import Image
from io import BytesIO
from flask import Flask
from keras.models import load_model

# import os
 
 
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import pandas as pd
# import random
# import ntpath

# from sklearn.model_selection import train_test_split
# from keras.models import Sequential
# from keras.optimizers import Adam
# from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
# from sklearn.utils import shuffle

sio = socketio.Server()

app = Flask(__name__) #'__main__'

def img_preprocess(img):
    img = img[55:130, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))#to match nvidia model input
    img = img/255
    return img

@sio.on('telemetry')
def telemetry(sid, data):
    image = Image.open(BytesIO(base64.b64decode(data["image"])))
    image = np.asarray(image)
    image = img_preprocess(image)  
    image = np.array([image])
    steering_angle = float(model.predict(image))
    send_control(steering_angle, 1.0)

@sio.on('connect') #mesasge, disconnect
def connect(sid, environ):
    print("connect")
    send_control(0, 0)

def send_control(steering_angle, throttle):
    sio.emit("steer", data = {
        'steering_angle': steering_angle.__str__(), 
        'throttle': throttle.__str__()
    })
    
if __name__ == '__main__':
    model = load_model('model.h5')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

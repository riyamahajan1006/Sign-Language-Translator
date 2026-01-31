import os
import cv2
import numpy as np
import mediapipe as mp
from tensorflow import keras
from flask import Flask,request,jsonify,render_template,Response
from keras.models import Sequential, load_model
from keras.layers import Conv2D,Dense,Flatten,MaxPooling2D,Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from werkzeug.utils import secure_filename
import warnings
warnings.filterwarnings("ignore")

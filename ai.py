# Test core libraries
import cv2
import sklearn
import tensorflow as tf
import numpy as np
print ("OpenCV version:", cv2.__version__)
print ("Scikit-learn version:", sklearn.__version__)
print ("TensorFlow version:", tf.__version__)
print ("GPU available:", tf.config.list_physical_devices('GPU'))
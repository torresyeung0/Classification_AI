import tensorflow as tf
import numpy as np
from PIL import Image
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

# Load our tensorflow model
model_path = os.path.join(script_dir, 'model.tflite')
interpertor = tf.lite.Interpreter(model_path=model_path)
interpertor.allocate_tensors()
# Load our labels/list of classes
labels_path = os.path.join(script_dir, 'labels.txt')
with open(labels_path, 'r') as f:
    labels = [line.strip() for line in f.readlines]

#Load and preprocess the image
def load_and_preprocess_image(image_path):
    img = Image.open(image_path).resize(224,224)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img.array, axis = 0)
    img_array /= 255.0
    return img_array
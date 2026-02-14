import numpy as np
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image

IMG_SIZE = (224, 224)

def preprocess_image(file):
    img = Image.open(file.stream).convert("RGB")
    img = img.resize(IMG_SIZE)

    array = keras_image.img_to_array(img) / 255.0
    array = np.expand_dims(array, axis=0)
    return array

def preprocess_image_with_original(file):
    original_img = Image.open(file.stream).convert("RGB")
    original_size = original_img.size

    img_resized = original_img.resize(IMG_SIZE)
    array = keras_image.img_to_array(img_resized) / 255.0
    array = np.expand_dims(array, axis=0)

    return array, original_size, original_img
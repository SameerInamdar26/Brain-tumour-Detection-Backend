import threading
import tensorflow as tf

_model = None
_model_lock = threading.Lock()

MODEL_PATH = "models/brain_tumour_model.h5"

def load_model_once():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return _model

def load_model():
    return load_model_once()

def predict_image(model, img_array):
    return model.predict(img_array)
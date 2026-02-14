from tensorflow.keras.models import load_model as keras_load_model
import threading

_model = None
_model_lock = threading.Lock()

MODEL_PATH = "models/brain_tumour_model.h5"

def load_model_once():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = keras_load_model(MODEL_PATH)
    return _model

def load_model():
    return load_model_once()

def predict_image(model, img_array):
    preds = model.predict(img_array)
    return preds
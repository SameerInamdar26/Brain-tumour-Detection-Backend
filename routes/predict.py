from flask import Blueprint, request, jsonify
import numpy as np
from services.preprocessing import preprocess_image
from services.inference import load_model, predict_image

predict_bp = Blueprint("predict", __name__)

model = load_model()

CATEGORIES = ["glioma", "meningioma", "notumor", "pituitary"]

@predict_bp.route("", methods=["POST"])
@predict_bp.route("/", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]

        img_array = preprocess_image(file)

        preds = predict_image(model, img_array)

        if isinstance(preds, list):
            preds = np.array(preds)

        preds = preds[0] if hasattr(preds, "ndim") and preds.ndim > 1 else preds

        predicted_class = int(np.argmax(preds))
        confidence = float(np.max(preds))

        label = CATEGORIES[predicted_class] if predicted_class < len(CATEGORIES) else "unknown"

        return jsonify({
            "prediction": label,
            "class_index": predicted_class,
            "confidence": confidence,
            "probs": preds.tolist()
        })
    except Exception as e:
        print("Predict error:", e)
        return jsonify({"error": str(e)}), 500
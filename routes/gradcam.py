from flask import Blueprint, request, jsonify, send_file
import io
import numpy as np
from services.preprocessing import preprocess_image_with_original
from services.inference import load_model, predict_image
from services.gradcam_service import make_gradcam_heatmap, overlay_gradcam

gradcam_bp = Blueprint("gradcam", __name__)

model = load_model()

@gradcam_bp.route("/", methods=["POST"])
def gradcam():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]

        img_array, original_size, original_img = preprocess_image_with_original(file)

        preds = predict_image(model, img_array)
        preds = preds[0] if hasattr(preds, "__len__") else preds
        pred_index = int(np.argmax(preds))

        heatmap = make_gradcam_heatmap(img_array, model, pred_index)
        cam_image = overlay_gradcam(original_img, heatmap, original_size)

        buf = io.BytesIO()
        cam_image.save(buf, format="PNG")
        buf.seek(0)

        return send_file(buf, mimetype="image/png")
    except Exception as e:
        print("GradCAM error:", e)
        return jsonify({"error": str(e)}), 500
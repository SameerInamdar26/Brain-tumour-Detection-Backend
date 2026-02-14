from flask import Flask
from flask_cors import CORS
from routes.predict import predict_bp
from routes.gradcam import gradcam_bp

def create_app():
    app = Flask(__name__)
    CORS(app)

    app.register_blueprint(predict_bp, url_prefix="/predict")
    app.register_blueprint(gradcam_bp, url_prefix="/gradcam")

    @app.route("/health", methods=["GET"])
    def health():
        return {"status": "ok"}

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=10000, debug=False)
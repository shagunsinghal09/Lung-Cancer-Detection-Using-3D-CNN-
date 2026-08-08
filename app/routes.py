import logging
from pathlib import Path

from flask import Blueprint, current_app, jsonify, render_template, request
from werkzeug.utils import secure_filename

from app.services.pipeline_service import PipelineService

bp = Blueprint("main", __name__)
logger = logging.getLogger(__name__)
service = PipelineService()


def _allowed(filename: str) -> bool:
    allowed = current_app.config["ALLOWED_EXTENSIONS"]
    return filename.endswith(".nii.gz") or filename.rsplit(".", 1)[-1].lower() in allowed


@bp.get("/")
def index():
    return render_template("index.html")


@bp.post("/predict")
def predict():
    if "scan" not in request.files:
        return jsonify({"error": "No scan file was provided."}), 400

    file = request.files["scan"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400
    if not _allowed(file.filename):
        return jsonify({"error": "Unsupported file format. Use .npy, .nii or .nii.gz"}), 400

    filename = secure_filename(file.filename)
    save_path = Path(current_app.config["UPLOAD_FOLDER"]) / filename
    file.save(save_path)

    try:
        result = service.predict(str(save_path))
        return jsonify(result)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed")
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

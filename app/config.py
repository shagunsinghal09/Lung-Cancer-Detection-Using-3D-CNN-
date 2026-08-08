from pathlib import Path


class Config:
    SECRET_KEY = "dev-secret-key"
    BASE_DIR = Path(__file__).resolve().parent.parent
    UPLOAD_FOLDER = BASE_DIR / "uploads"
    MAX_CONTENT_LENGTH = 512 * 1024 * 1024  # 512 MB
    ALLOWED_EXTENSIONS = {"npy", "nii", "nii.gz"}
    MODEL_CHECKPOINT = BASE_DIR / "models" / "fusion_classifier.pt"
    LOG_LEVEL = "INFO"

import logging
from pathlib import Path

from flask import Flask

from app.config import Config


def create_app(config_class=Config) -> Flask:
    app = Flask(__name__, static_folder="../static", template_folder="../templates")
    app.config.from_object(config_class)

    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, app.config["LOG_LEVEL"], logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    from app.routes import bp

    app.register_blueprint(bp)
    return app

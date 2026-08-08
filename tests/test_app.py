import io

import numpy as np
from app import create_app


def test_predict_with_npy_upload():
    app = create_app()
    client = app.test_client()

    arr = np.random.rand(32, 32, 32).astype("float32")
    buf = io.BytesIO()
    np.save(buf, arr)
    buf.seek(0)

    response = client.post(
        "/predict",
        data={"scan": (buf, "sample.npy")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert "label" in payload
    assert "probability" in payload
    assert "confidence" in payload


def test_predict_rejects_missing_file():
    app = create_app()
    client = app.test_client()

    response = client.post("/predict", data={}, content_type="multipart/form-data")
    assert response.status_code == 400

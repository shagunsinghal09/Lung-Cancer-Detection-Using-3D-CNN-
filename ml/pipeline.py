from __future__ import annotations

import numpy as np

from ml.classification.classifier import FusionClassifier
from ml.fusion.feature_fusion import fuse_features
from ml.preprocessing.ct_preprocessing import preprocess_ct
from ml.radiomics.radiomics_extractor import extract_radiomics_features


class LungCancerPipeline:
    def __init__(self, seed: int = 42) -> None:
        np.random.seed(seed)
        self._torch = None
        self.encoder = None
        try:
            import torch
            from ml.models.cnn3d import CNN3DEncoder

            torch.manual_seed(seed)
            self._torch = torch
            self.encoder = CNN3DEncoder(out_features=16)
            self.encoder.eval()
        except Exception:
            self._torch = None
            self.encoder = None
        self.classifier = FusionClassifier()

    def _cnn_features(self, volume: np.ndarray) -> np.ndarray:
        if self._torch is None or self.encoder is None:
            return np.array(
                [
                    float(np.mean(volume)),
                    float(np.std(volume)),
                    float(np.min(volume)),
                    float(np.max(volume)),
                ],
                dtype=np.float32,
            )

        tensor = self._torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)
        with self._torch.no_grad():
            features = self.encoder(tensor).squeeze(0).cpu().numpy()
        return features.astype(np.float32)

    def predict(self, path: str) -> dict:
        volume = preprocess_ct(path)
        cnn_features = self._cnn_features(volume)
        radiomics_features = extract_radiomics_features(volume)
        fused = fuse_features(cnn_features, radiomics_features)
        result = self.classifier.predict(fused)
        result["feature_vector_size"] = int(fused.shape[0])
        return result

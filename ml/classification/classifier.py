import numpy as np


class FusionClassifier:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def predict_proba(self, features: np.ndarray) -> float:
        score = float(np.mean(features))
        probability = 1 / (1 + np.exp(-5 * (score - 0.5)))
        return float(np.clip(probability, 0.0, 1.0))

    def predict(self, features: np.ndarray) -> dict:
        prob = self.predict_proba(features)
        label = "Suspicious" if prob >= self.threshold else "Non-suspicious"
        confidence = max(prob, 1 - prob)
        return {"label": label, "probability": prob, "confidence": confidence}

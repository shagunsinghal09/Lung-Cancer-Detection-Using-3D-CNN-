import numpy as np


def fuse_features(cnn_features: np.ndarray, radiomics_features: np.ndarray) -> np.ndarray:
    cnn = np.asarray(cnn_features, dtype=np.float32).ravel()
    radio = np.asarray(radiomics_features, dtype=np.float32).ravel()
    return np.concatenate([cnn, radio], axis=0)

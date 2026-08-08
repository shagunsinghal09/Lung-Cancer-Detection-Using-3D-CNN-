from __future__ import annotations

import numpy as np


def simple_radiomics_features(volume: np.ndarray) -> np.ndarray:
    return np.array(
        [
            float(np.mean(volume)),
            float(np.std(volume)),
            float(np.min(volume)),
            float(np.max(volume)),
            float(np.percentile(volume, 25)),
            float(np.percentile(volume, 50)),
            float(np.percentile(volume, 75)),
        ],
        dtype=np.float32,
    )


def extract_radiomics_features(volume: np.ndarray) -> np.ndarray:
    try:
        from radiomics import firstorder

        extractor = firstorder.RadiomicsFirstOrder(imageArray=volume, maskArray=np.ones_like(volume))
        extractor.enableAllFeatures()
        result = extractor.execute()
        values = [float(v) for _, v in sorted(result.items()) if isinstance(v, (int, float))]
        if values:
            return np.array(values, dtype=np.float32)
    except Exception:
        pass

    return simple_radiomics_features(volume)

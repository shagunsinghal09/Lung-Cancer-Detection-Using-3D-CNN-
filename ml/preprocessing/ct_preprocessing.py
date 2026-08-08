from __future__ import annotations

from pathlib import Path

import numpy as np


def _center_crop_or_pad(volume: np.ndarray, target_shape=(64, 64, 64)) -> np.ndarray:
    out = np.zeros(target_shape, dtype=np.float32)

    src_slices = []
    dst_slices = []
    for current, target in zip(volume.shape, target_shape):
        if current >= target:
            start_src = (current - target) // 2
            src_slices.append(slice(start_src, start_src + target))
            dst_slices.append(slice(0, target))
        else:
            start_dst = (target - current) // 2
            src_slices.append(slice(0, current))
            dst_slices.append(slice(start_dst, start_dst + current))

    out[tuple(dst_slices)] = volume[tuple(src_slices)]
    return out


def load_ct_volume(path: str) -> np.ndarray:
    file_path = Path(path)
    if file_path.suffix == ".npy":
        volume = np.load(path)
    elif file_path.name.endswith(".nii") or file_path.name.endswith(".nii.gz"):
        try:
            import nibabel as nib
        except ImportError as exc:
            raise RuntimeError("nibabel is required to read NIfTI files.") from exc
        volume = np.asarray(nib.load(path).get_fdata())
    else:
        raise ValueError("Unsupported scan format")

    if volume.ndim != 3:
        raise ValueError("Input volume must be 3D")

    return volume.astype(np.float32)


def normalize_ct(volume: np.ndarray) -> np.ndarray:
    clipped = np.clip(volume, -1000, 400)
    normalized = (clipped + 1000.0) / 1400.0
    return normalized


def preprocess_ct(path: str, target_shape=(64, 64, 64)) -> np.ndarray:
    volume = load_ct_volume(path)
    volume = normalize_ct(volume)
    return _center_crop_or_pad(volume, target_shape=target_shape)

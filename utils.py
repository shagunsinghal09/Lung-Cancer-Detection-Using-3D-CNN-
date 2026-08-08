import numpy as np
from scipy.ndimage import zoom, binary_fill_holes
from skimage import measure

def resample_volume(vol, current_spacing, target_spacing=(1.0,1.0,1.0)):
    # vol: ndarray (Z, H, W)
    csx, csy, csz = current_spacing  # x->width, y->height, z->slice
    tsx, tsy, tsz = target_spacing
    # zoom order corresponds to vol axes (z,y,x): factors = (dz, dy, dx)
    zoom_factors = (csz/tsz, csy/tsy, csx/tsx)
    vol_rs = zoom(vol, zoom_factors, order=1)
    return vol_rs

def normalize_window(vol, lo=-1000, hi=400):
    vol = np.clip(vol, lo, hi)
    vol = (vol - lo) / (hi - lo)
    return vol.astype(np.float32)

def simple_normalize_0_255(vol):
    v = vol.astype(np.float32)
    v = (v - v.min()) / (v.max() - v.min() + 1e-8)
    return v.astype(np.float32)

def connected_components(mask, min_voxels=10):
    # mask: boolean ndarray (Z,H,W)
    labeled = measure.label(mask)
    props = measure.regionprops(labeled)
    out = []
    for p in props:
        if p.area >= min_voxels:
            out.append({'label': p.label, 'bbox': p.bbox, 'area': p.area, 'coords': p.coords})
    return labeled, out

def component_diameter_mm(coords, spacing):
    # coords: Nx3 array (z,y,x). spacing: (sx, sy, sz) -> x,y,z (mm)
    coords_mm = coords.astype(float).copy()
    # convert to (x_mm, y_mm, z_mm)
    coords_mm[:, 2] *= spacing[0]
    coords_mm[:, 1] *= spacing[1]
    coords_mm[:, 0] *= spacing[2]
    mins = coords_mm.min(axis=0)
    maxs = coords_mm.max(axis=0)
    diag = np.linalg.norm(maxs - mins)
    return float(diag)

def suggest_t_stage(diameter_mm):
    if diameter_mm == 0:
        return 'No nodule detected'
    d = diameter_mm
    if d <= 10: return 'T1a'
    if d <= 20: return 'T1b'
    if d <= 30: return 'T1c'
    if d <= 50: return 'T2'
    if d <= 70: return 'T3'
    return 'T4'

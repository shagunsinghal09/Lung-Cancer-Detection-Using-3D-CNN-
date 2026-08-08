import argparse, numpy as np, torch
from utils import connected_components, component_diameter_mm, suggest_t_stage
from scipy.ndimage import binary_fill_holes, zoom
import json

# ----------------------------------------------------
# 🔧 Add simple_normalize_0_255 here
# ----------------------------------------------------
def simple_normalize_0_255(vol):
    """Normalize volume (3D numpy array) to 0–1 range."""
    vol = vol.astype(np.float32)
    vmin, vmax = vol.min(), vol.max()
    if vmax - vmin > 0:
        vol = (vol - vmin) / (vmax - vmin)
    else:
        vol[:] = 0
    return vol

# ----------------------------------------------------
# 🔧 Add resample_volume helper
# ----------------------------------------------------
def resample_volume(vol, current_spacing=(1.0, 1.0, 5.0), target_spacing=(1.0, 1.0, 1.0)):
    """Resample a 3D volume to new voxel spacing using linear interpolation."""
    zoom_factors = [cs / ts for cs, ts in zip(current_spacing, target_spacing)]
    vol_resampled = zoom(vol, zoom_factors, order=1)
    return vol_resampled

# ----------------------------------------------------
# Existing Wrapper class
# ----------------------------------------------------
class Wrapper:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        try:
            self.model = torch.jit.load(model_path, map_location=self.device)
            self.model.eval()
        except Exception:
            from train_unet import UNet3D
            m = UNet3D(in_ch=1, out_ch=1, base=8)
            sd = torch.load(model_path, map_location=self.device)
            m.load_state_dict(sd)
            m.to(self.device).eval()
            self.model = m

    def predict(self, vol):
        # vol: numpy (Z,H,W) normalized to 0..1
        import torch
        z, h, w = vol.shape
        tz, th, tw = 32, 64, 64
        pv = np.zeros((tz, th, tw), dtype=np.float32)
        dz = min(tz, z); startz = max(0, (z - dz) // 2); pstz = (tz - dz) // 2
        dh = min(th, h); starth = max(0, (h - dh) // 2); psth = (th - dh) // 2
        dw = min(tw, w); startw = max(0, (w - dw) // 2); pstw = (tw - dw) // 2
        pv[pstz:pstz + dz, psth:psth + dh, pstw:pstw + dw] = vol[startz:startz + dz, starth:starth + dh, startw:startw + dw]
        x = torch.from_numpy(pv).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(x)
        pred = out.squeeze().cpu().numpy()
        pred_rs = zoom(pred, (z / pred.shape[0], h / pred.shape[1], w / pred.shape[2]), order=1)
        return pred_rs


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--volume', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--min_voxels', type=int, default=10)
    args = p.parse_args()

    d = np.load(args.volume)
    vol = d['volume'].astype(np.float32)
    spacing = d.get('spacing', np.array([1.0, 1.0, 1.0]))
    wr = Wrapper(args.model, device='cpu')
    pred = wr.predict(vol)
    mask = pred > 0.5
    mask = binary_fill_holes(mask)
    labeled, comps = connected_components(mask, min_voxels=args.min_voxels)
    nodules = []
    largest = 0.0
    for c in comps:
        dia = component_diameter_mm(c['coords'], spacing=(spacing[0], spacing[1], spacing[2]))
        nodules.append({'label': c['label'], 'area': c['area'], 'diameter_mm': dia})
        if dia > largest:
            largest = dia
    out = {
        'nodule_count': len(nodules),
        'nodules': nodules,
        'largest_diameter_mm': largest,
        't_stage_suggestion': suggest_t_stage(largest)
    }
    print(json.dumps(out, indent=2))

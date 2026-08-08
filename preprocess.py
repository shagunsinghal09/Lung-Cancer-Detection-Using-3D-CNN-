import os, argparse, numpy as np
from PIL import Image
from utils import resample_volume, simple_normalize_0_255

def load_png_stack(folder):
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith('.png')])
    if len(files) == 0:
        raise ValueError('No PNG files found in folder: ' + folder)
    imgs = [np.array(Image.open(os.path.join(folder, f)).convert('L')) for f in files]
    vol = np.stack(imgs, axis=0).astype(np.float32)
    return vol

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input_dir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--spacing', nargs=3, type=float, required=False, default=(1.0,1.0,5.0),
                   help='current spacing as sx sy sz in mm (pixel width, pixel height, slice thickness)')
    p.add_argument('--target_spacing', nargs=3, type=float, default=(1.0,1.0,1.0))
    args = p.parse_args()

    vol = load_png_stack(args.input_dir)
    vol = simple_normalize_0_255(vol)
    vol_rs = resample_volume(vol, current_spacing=tuple(args.spacing), target_spacing=tuple(args.target_spacing))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, volume=vol_rs, spacing=np.array(args.target_spacing))
    print('Saved', args.out, 'shape=', vol_rs.shape)

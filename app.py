from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile, io, os, base64
import numpy as np
from zipfile import ZipFile
from PIL import Image

# Import helpers from inference.py
from inference import (
    Wrapper,
    simple_normalize_0_255,
    resample_volume,
    connected_components,
    component_diameter_mm,
    suggest_t_stage
)

# ------------------ Setup FastAPI ------------------
app = FastAPI(title="Lung Nodule Detection API")

# Allow Streamlit frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:8501"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Load Model ------------------
MODEL_PATH = "sample/unet_dummy_ts.pt"
wr = Wrapper(MODEL_PATH, device='cpu')

# ------------------ API Endpoint ------------------
@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    fn = file.filename
    data = await file.read()
    tmpdir = tempfile.TemporaryDirectory()

    # Handle ZIP of PNGs
    if fn.lower().endswith('.zip'):
        z = ZipFile(io.BytesIO(data))
        z.extractall(tmpdir.name)
        files = sorted([
            os.path.join(tmpdir.name, f)
            for f in os.listdir(tmpdir.name)
            if f.lower().endswith('.png')
        ])
        imgs = [np.array(Image.open(f).convert('L')) for f in files]
        vol = np.stack(imgs, axis=0).astype(np.float32)
        vol = simple_normalize_0_255(vol)
        vol_rs = resample_volume(vol, current_spacing=(1.0, 1.0, 5.0), target_spacing=(1.0, 1.0, 1.0))

    # Handle NPZ
    elif fn.lower().endswith('.npz'):
        p = os.path.join(tmpdir.name, fn)
        with open(p, 'wb') as f:
            f.write(data)
        d = np.load(p)
        keys = list(d.keys())
        preferred = ['volume', 'vol', 'image', 'images', 'data', 'arr_0', 'mask']
        chosen = next((k for k in preferred if k in keys), None) or (keys[0] if keys else None)
        if chosen is None:
            return JSONResponse(
                {'error': 'NPZ archive is empty — no arrays found.'},
                status_code=400,
            )
        arr = d[chosen]
        if arr.ndim != 3:
            return JSONResponse(
                {'error': f"NPZ key '{chosen}' has shape {arr.shape}; expected a 3D volume (Z,H,W). Available keys: {keys}"},
                status_code=400,
            )
        vol_rs = arr.astype(np.float32)

    else:
        return JSONResponse({'error': 'Upload .zip of PNGs or .npz volume'}, status_code=400)

    # Prediction
    pred = wr.predict(vol_rs)
    mask = (pred > 0.5).astype(np.uint8)
    labeled, comps = connected_components(mask, min_voxels=10)
    nodules, largest = [], 0.0
    for c in comps:
        dia = component_diameter_mm(c['coords'], spacing=(1.0, 1.0, 1.0))
        nodules.append({'label': c['label'], 'area': c['area'], 'diameter_mm': dia})
        largest = max(largest, dia)

    # Central slice preview (mask)
    z = mask.shape[0] // 2
    img = Image.fromarray((mask[z] * 255).astype('uint8'))
    buf = io.BytesIO()
    img.save(buf, 'PNG')
    mask_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Central slice preview (uploaded CT volume)
    vol_slice = vol_rs[vol_rs.shape[0] // 2]
    vmin, vmax = float(vol_slice.min()), float(vol_slice.max())
    if vmax - vmin < 1e-8:
        vol_norm = np.zeros_like(vol_slice, dtype=np.uint8)
    else:
        vol_norm = ((vol_slice - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
    vol_img = Image.fromarray(vol_norm)
    vbuf = io.BytesIO()
    vol_img.save(vbuf, 'PNG')
    vol_b64 = base64.b64encode(vbuf.getvalue()).decode('utf-8')

    cancer_detected = len(nodules) > 0
    verdict = 'Cancer Detected' if cancer_detected else 'No Cancer Detected'

    return JSONResponse({
        'nodule_count': len(nodules),
        'nodules': nodules,
        'largest_diameter_mm': largest,
        't_stage_suggestion': suggest_t_stage(largest),
        'cancer_detected': cancer_detected,
        'verdict': verdict,
        'mask_central_png_base64': mask_b64,
        'volume_central_png_base64': vol_b64
    })


# ------------------ Run the API ------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)

# Lung Cancer Detection — 3D U-Net Demo (Educational)

**Purpose:** A compact, runnable demo project that shows how to preprocess a folder of PNG slices into a 3D volume, run a lightweight 3D U-Net segmentation model, postprocess detected nodules, and serve a simple FastAPI endpoint with a React demo UI.

**Important:** This is a research/demo tool **only** — not for clinical use.

## What's included
- `preprocess.py` : convert PNG stack -> resampled .npz volume
- `utils.py` : helper functions (resampling, normalization, connected components, diameter)
- `train_unet.py` : lightweight training script (for small local experiments)
- `inference.py` : run model on .npz -> masks + nodule info + T-stage suggestion
- `backend/app.py` : FastAPI server (endpoint `/predict` accepts .npz or .zip of PNGs)
- `frontend/App.jsx` : single-file React component (drop into a Vite/CRA app)
- `sample/` : a tiny synthetic example `.npz` to test inference without training
- `requirements.txt`

## Quick run (local test using the included synthetic sample)

1. Create and activate Python environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Run inference on synthetic sample to see pipeline output:
   ```bash
   python inference.py --volume sample/synthetic_volume.npz --model sample/unet_dummy_ts.pt
   ```
   This will print a JSON result with detected nodules (the included model is a tiny TorchScript model that produces a synthetic mask for demo).

3. Run the FastAPI backend for demo:
   ```bash
   uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
   ```
   Then open the React demo (see `frontend/App.jsx`) or call `/predict` with `curl`/Postman.

## If you want to actually train
- `train_unet.py` includes a minimal training loop and saving. For real training, prepare `.npz` volumes and paired `_mask.npz` files, and adjust patch size / augmentations / GPU settings.

## Notes
- The included TorchScript model is intentionally tiny to allow quick CPU runs for testing the pipeline.
- For a production/research-quality system, replace with full LIDC/LUNA preprocessed datasets, heavier models, multi-scale inference, extensive augmentations, and clinician validation.

---
If you'd like, I can now:
- Expand the dataset loader to create 64x64x64 patches with on-the-fly augmentation.
- Provide a script that converts DICOM -> PNG stack (with HU preservation).
- Build and upload a full trained checkpoint (if you provide data or allow me to use public data).

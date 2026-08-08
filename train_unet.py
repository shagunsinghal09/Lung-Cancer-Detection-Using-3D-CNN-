import os, argparse, numpy as np
from glob import glob
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# =====================================================
# Utility: create synthetic data if dataset folder empty
# =====================================================
def ensure_sample_data(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    files = glob(os.path.join(data_dir, "vol*.npz"))
    if len(files) == 0:
        print(f"⚠️ No data found in {data_dir} → Creating synthetic volumes...")
        for i in range(6):  # create 6 volumes instead of 3
            vol = np.random.rand(32, 64, 64).astype(np.float32)
            mask = (vol > 0.7).astype(np.float32)
            np.savez(os.path.join(data_dir, f"vol_{i}.npz"), volume=vol)      # ✅ correct key
            np.savez(os.path.join(data_dir, f"vol_{i}_mask.npz"), mask=mask)  # ✅ correct key
        print("✅ Synthetic data created.")
    else:
        print(f"📂 Found {len(files)} existing .npz volume files in {data_dir}")

# =====================================================
# 3D Double Convolution Block
# =====================================================
class DoubleConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

# =====================================================
# 3D U-Net Architecture
# =====================================================
class UNet3D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=8):
        super().__init__()
        self.enc1 = DoubleConv3D(in_ch, base)
        self.pool = nn.MaxPool3d(2)
        self.enc2 = DoubleConv3D(base, base * 2)
        self.up2 = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
        self.dec2 = DoubleConv3D(base * 2, base)
        self.outc = nn.Conv3d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        d2 = self.up2(e2)
        diffZ = e1.size(2) - d2.size(2)
        diffY = e1.size(3) - d2.size(3)
        diffX = e1.size(4) - d2.size(4)
        d2 = F.pad(d2, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        d2 = torch.cat([d2, e1], dim=1)
        d1 = self.dec2(d2)
        return torch.sigmoid(self.outc(d1))

# =====================================================
# Dataset Loader
# =====================================================
# =====================================================
# Dataset Loader (Fixed & Flexible)
# =====================================================
class VolDataset(Dataset):
    def __init__(self, data_dir):
        # Accept both "vol0.npz" and "vol_0.npz"
        self.files = sorted(glob(os.path.join(data_dir, "vol*.npz")))
        # Ignore mask files (only keep main volumes)
        self.files = [f for f in self.files if "_mask" not in f]

        if len(self.files) == 0:
            raise RuntimeError(f"No volume files found in {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        d = np.load(path)

        # Ensure compatibility with different npz formats
        if "volume" in d:
            vol = d["volume"].astype(np.float32)
        else:
            # Fallback if no key (e.g. plain array saved with np.savez)
            vol = list(d.values())[0].astype(np.float32)

        maskp = path.replace(".npz", "_mask.npz")
        if os.path.exists(maskp):
            mask_data = np.load(maskp)
            mask = mask_data["mask"].astype(np.float32) if "mask" in mask_data else list(mask_data.values())[0].astype(np.float32)
        else:
            mask = (vol > vol.mean()).astype(np.float32)

        vol = np.expand_dims(vol, 0)
        mask = np.expand_dims(mask, 0)
        return torch.from_numpy(vol), torch.from_numpy(mask)

# =====================================================
# Combined Dice + BCE Loss
# =====================================================
class DiceBCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        p, t = pred.view(-1), target.view(-1)
        b = self.bce(p, t)
        inter = (p * t).sum()
        dice = (2 * inter + 1e-6) / (p.sum() + t.sum() + 1e-6)
        return b + (1 - dice)

# =====================================================
# Training Script
# =====================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--bs", type=int, default=1)
    p.add_argument("--save", default="models/unet_demo.pt")
    args = p.parse_args()

    ensure_sample_data(args.data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = VolDataset(args.data_dir)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=0)

    model = UNet3D(in_ch=1, out_ch=1, base=8).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = DiceBCE()
    best = float("inf")

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    print(f"🚀 Starting training on {len(ds)} samples ({device})")

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for x, m in dl:
            x, m = x.to(device), m.to(device)
            pred = model(x)
            loss = loss_fn(pred, m)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()

        avg = total / len(dl)
        print(f"📘 Epoch {epoch + 1}/{args.epochs} → avg_loss={avg:.4f}")

        if avg < best:
            best = avg
            torch.save(model.state_dict(), args.save)
            try:
                model.eval()
                example = torch.randn(1, 1, 32, 64, 64).to(device)
                traced = torch.jit.trace(model, example)
                traced.save(args.save.replace(".pt", "_ts.pt"))
                print("💾 Model and TorchScript exported successfully.")
            except Exception as e:
                print("⚠️ TorchScript export failed:", e)

    print(f"✅ Training complete. Best loss = {best:.4f}")

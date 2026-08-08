import torch
import torch.nn as nn
import os

os.makedirs("sample", exist_ok=True)

# Dummy 3D model
class DummyUNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 1, 3, padding=1)
    def forward(self, x):
        return torch.sigmoid(self.conv(x))

model = DummyUNet3D()
scripted_model = torch.jit.script(model)
scripted_model.save("sample/unet_dummy_ts.pt")

print("✅ Dummy 3D model created at sample/unet_dummy_ts.pt")


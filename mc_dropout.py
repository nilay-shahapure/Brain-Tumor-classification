
import os
import numpy as np
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm


def mc_dropout_predict(model, x, T=20):
    model.train()
    probs = []
    with torch.no_grad():
        for _ in range(T):
            logits = model(x)
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    probs = np.stack(probs, axis=0)
    return probs.mean(axis=0), probs.std(axis=0)

# Config
data_dir   = "data"
out_dir    = "checkpoints"
model_path = os.path.join(out_dir, "best.pth")
batch_size = 32
mc_T       = 20

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.resnet18(pretrained=False)
nfeat = model.fc.in_features
model.fc = torch.nn.Sequential(torch.nn.Dropout(0.5),
                               torch.nn.Linear(nfeat, 2))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device).eval()

# Validation loader (same transforms used earlier)
val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225]),
])
val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_tf)
val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

# Run MC-Dropout
all_std = []
for imgs, _ in tqdm(val_ld, desc="MC-Dropout"):
    _, std = mc_dropout_predict(model, imgs.to(device), T=mc_T)
    all_std.extend(std.max(axis=1))

# Save
np.save(os.path.join(out_dir, "uncertainties.npy"), np.array(all_std))
print("Saved uncertainties.npy →", os.path.join(out_dir, "uncertainties.npy"))


# CONFIG 
data_dir   = "data"           # must contain train/yes, train/no, val/yes, val/no
out_dir    = "checkpoints"    # where to save models
epochs     = 20
batch_size = 32
lr         = 1e-4
use_cuda   = True             # set False to force CPU

# ─── IMPORTS ─────────────────────────────────────────────────────────
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# ─── SETUP ────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu")
os.makedirs(out_dir, exist_ok=True)

# ─── DATA ─────────────────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
train_ds = datasets.ImageFolder(os.path.join(data_dir,"train"), transform=train_tf)
val_ds   = datasets.ImageFolder(os.path.join(data_dir,"val"  ), transform=val_tf)
train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
val_ld   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)

# ─── MODEL + OPTIMIZER ───────────────────────────────────────────────
model = models.resnet18(pretrained=True)
nfeat = model.fc.in_features
model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(nfeat, len(train_ds.classes)))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# ─── TRAIN/VALID LOOP ─────────────────────────────────────────────────
best_acc, best_file = 0.0, None

for epoch in range(1, epochs+1):
    # train
    model.train()
    train_loss = 0.0
    for imgs, labels in tqdm(train_ld, desc=f"Epoch {epoch} Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * imgs.size(0)
    train_loss /= len(train_ds)

    # validate
    model.eval()
    val_loss, correct = 0.0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(val_ld, desc=f"Epoch {epoch} Val", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            val_loss += criterion(outputs, labels).item() * imgs.size(0)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
    val_loss /= len(val_ds)
    val_acc = correct / len(val_ds)

    print(f"Epoch {epoch}/{epochs}  "
          f"train_loss {train_loss:.4f}  "
          f"val_loss {val_loss:.4f}  "
          f"val_acc {val_acc:.4f}")

    # checkpoint
    if val_acc > best_acc:
        best_acc = val_acc
        best_file = os.path.join(out_dir, "best.pth")
        torch.save(model.state_dict(), best_file)
    scheduler.step()

# final save
final_file = os.path.join(out_dir, "final.pth")
torch.save(model.state_dict(), final_file)
print(f"Training done. Best acc {best_acc:.4f} → {best_file}")
print(f"Final model saved to → {final_file}")

#---------------------------------------------------------------------------------------------GRADCAM AND EMBEDDING ANALYSIS-------------------------------------------------------------------------------------

# ─── CONFIG ───────────────────────────────────────────────────────────
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# Where your trained model lives:
model_path = "checkpoints/best.pth"
data_dir   = "data"
out_dir    = "checkpoints"
os.makedirs(out_dir, exist_ok=True)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── LOAD MODEL ───────────────────────────────────────────────────────
# Re-build your architecture
model = models.resnet18(pretrained=False)
# replace fc the same way you did in training
nfeat = model.fc.in_features
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),
    torch.nn.Linear(nfeat, len(os.listdir(f"{data_dir}/train/yes"))>0 and 2 or 1)
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device).eval()

# ─── DATA LOADER ──────────────────────────────────────────────────────
tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225]),
])
val_ds = datasets.ImageFolder(f"{data_dir}/val", transform=tf)
val_ld = DataLoader(val_ds, batch_size=4, shuffle=True, num_workers=1)

# ─── CORRECTED Grad-CAM HELPER ────────────────────────────────────────
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# unnormalize helper
IM_MEAN = np.array([0.485,0.456,0.406])
IM_STD  = np.array([0.229,0.224,0.225])
def unnormalize(x):
    img = x.cpu().numpy().transpose(1,2,0)
    img = (img * IM_STD) + IM_MEAN
    return np.clip(img, 0, 1)

def generate_gradcam_overlay(model, x, target_layer):
    # instantiate without use_cuda kw
    try:
        cam = GradCAM(model=model, target_layers=[target_layer])
    except TypeError:
        # some versions expect 'device'
        cam = GradCAM(model=model, target_layers=[target_layer], device=device)
    # forward & get map
    grayscale = cam(input_tensor=x)[0]
    img = unnormalize(x.squeeze(0))
    # clear hooks (if supported)
    if hasattr(cam, "clear_hooks"):
        cam.clear_hooks()
    return show_cam_on_image(img, grayscale, use_rgb=True)

# ─── GENERATE & SAVE Grad-CAM OVERLAYS ───────────────────────────────
print("Saving Grad-CAM overlays …")
imgs, _ = next(iter(val_ld))
for i, img in enumerate(imgs):
    x = img.unsqueeze(0).to(device)
    overlay = generate_gradcam_overlay(
        model, x, target_layer=model.layer4[-1].conv2
    )
    plt.imsave(f"{out_dir}/gradcam_{i}.png", overlay)
print("… done")

# ─── UMAP / t-SNE EMBEDDING ───────────────────────────────────────────
print("Computing UMAP/t-SNE …")
# build backbone up to avgpool
backbone = torch.nn.Sequential(
    model.conv1, model.bn1, model.relu, model.maxpool,
    model.layer1, model.layer2, model.layer3, model.layer4,
    model.avgpool, torch.nn.Flatten()
).to(device)

feats, labs = [], []
with torch.no_grad():
    for imgs, lbl in val_ld:
        f = backbone(imgs.to(device)).cpu().numpy()
        feats.append(f); labs.append(lbl.numpy())
feats = np.vstack(feats)
labs  = np.hstack(labs)

# UMAP vs t-SNE fallback
try:
    from umap import UMAP
    reducer = UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    method = "UMAP"
except ImportError:
    from sklearn.manifold import TSNE
    reducer = TSNE(n_components=2, random_state=42)
    method = "t-SNE"

emb = reducer.fit_transform(feats)
plt.figure(figsize=(7,7))
df = {'x': emb[:,0], 'y': emb[:,1], 'label': labs}
sns.scatterplot(data=df, x='x', y='y', hue='label',
                palette='viridis', s=12)
plt.title(method)
plt.savefig(f"{out_dir}/embedding_{method}.png", dpi=300)
plt.close()
print("Embedding saved")

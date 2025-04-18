# scripts/extract_features.py
import os
import logging
import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
import cv2

# ——— Logging setup ———
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# ——— 1) Load ResNet50 with new weights API ———
model = torchvision.models.resnet50(
    weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1
)
model.eval()
feat_extractor = torch.nn.Sequential(*list(model.children())[:-1])

# ——— 2) Preprocessing + CLAHE for lighting robustness ———
def preprocess(img: Image.Image) -> torch.Tensor:
    # PIL→OpenCV BGR
    arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    pil = Image.fromarray(rgb)

    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std =[0.229,0.224,0.225])
    ])
    return tf(pil)

# ——— 3) Read CSV & extract features with TTA rotations ———
df = pd.read_csv("data/bottles.csv", dtype={"id": str})
feats_list, ids = [], []

for _, row in df.iterrows():
    img_id = row["id"]
    path   = f"data/images/{img_id}.jpg"
    if not os.path.isfile(path):
        logging.warning(f"{img_id}: missing image, skipping.")
        continue

    try:
        img = Image.open(path).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        logging.error(f"{img_id}: cannot open image ({e}), skipping.")
        continue

    # Test‐time augmentation: small rotations
    angles = [0, -10, 10]
    buffs = []
    for ang in angles:
        im_rot = img.rotate(ang, resample=Image.BILINEAR)
        x = preprocess(im_rot).unsqueeze(0)
        with torch.no_grad():
            f = feat_extractor(x).squeeze().cpu().numpy()
        buffs.append(f)
    feat_avg = np.mean(buffs, axis=0)

    feats_list.append(feat_avg.astype("float32"))
    ids.append(int(img_id))
    logging.info(f"{img_id}: feature extracted")

if not feats_list:
    raise RuntimeError("No features extracted! Check your images folder.")

# ——— 4) Stack, normalize, and save ———
feats = np.stack(feats_list)                      # (M,2048)
norms = np.linalg.norm(feats, axis=1, keepdims=True)
feats_norm = feats / norms                        # L2‑normalized

os.makedirs("data", exist_ok=True)
np.save("data/feats_norm.npy", feats_norm)        # for cosine sims
pd.DataFrame({"bottle_id": ids}) \
  .to_csv("data/id_map.csv", index=False)
logging.info(f"Saved feats_norm.npy ({feats_norm.shape}) + id_map.csv")

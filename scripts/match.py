# scripts/match.py
import numpy as np
import pandas as pd
import subprocess
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from PIL import Image
import cv2

# ——— 1) Load normalized features + metadata ———
feats_norm = np.load("data/feats_norm.npy")       # (N,2048)
id_map     = pd.read_csv("data/id_map.csv")        # column: bottle_id
bottles    = pd.read_csv("data/bottles.csv", index_col="id")

# ——— 2) Load ResNet50 for embedding ———
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.eval()
feat_extractor = torch.nn.Sequential(*list(model.children())[:-1])

# ——— 3) Same CLAHE preprocess as extract_features ———
def preprocess(img: Image.Image) -> torch.Tensor:
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

def extract_feat_norm(img_path: str) -> np.ndarray:
    """Embed + normalize a query image, with simple TTA rotations."""
    img = Image.open(img_path).convert("RGB")
    angles = [0, -10, 10]
    buffs = []
    for ang in angles:
        im_rot = img.rotate(ang, resample=Image.BILINEAR)
        x = preprocess(im_rot).unsqueeze(0)
        with torch.no_grad():
            f = feat_extractor(x).squeeze().cpu().numpy()
        buffs.append(f)
    avg = np.mean(buffs, axis=0)
    return (avg / np.linalg.norm(avg)).astype("float32")

def ocr_label(img_path: str) -> str:
    """One‐shot Tesseract CLI call."""
    try:
        res = subprocess.run(
            ["tesseract", img_path, "stdout"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
            text=True
        )
        return res.stdout.strip()
    except Exception:
        return ""

def match(query_path: str, top_k: int = 3):
    q = extract_feat_norm(query_path)        # (2048,)
    sims = feats_norm @ q                    # cosine similarities
    idxs = np.argsort(-sims)[:top_k]

    out = []
    for i in idxs:
        sim = float(sims[i])
        bid = int(id_map.iloc[i]["bottle_id"])
        info = bottles.loc[bid].to_dict()
        info.update(confidence=sim)
        out.append(info)
    return out

if __name__ == "__main__":
    import sys
    query = sys.argv[1]
    ocr   = ocr_label(query)

    for r in match(query, top_k=5):
        print(
            f"{r['name']}  "
            f"(confidence={r['confidence']:.4f}) → "
            f"MSRP ${r['avg_msrp']}, shelf ${r['shelf_price']}"
        )
        if ocr:
            print("OCR guess:", ocr)
        print("-" * 40)

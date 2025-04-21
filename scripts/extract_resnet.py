"""
Build 2048‑d ResNet50 embeddings for the 500 catalog photos
→ data/feats_norm.npy   (N,2048) L2‑normalised
"""
import numpy as np, pandas as pd, torch, cv2, os
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

# Load bottle catalog
df = pd.read_csv("data/bottles.csv", dtype={"id":str})
paths = [(i, f"data/images/{i}.jpg") for i in df["id"]]

# Load ResNet50 feature extractor
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.eval()
extract = torch.nn.Sequential(*list(model.children())[:-1])

# CLAHE preprocessing
def clahe(pil):
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    cl = cv2.createCLAHE(2.0,(8,8)).apply(l)
    return Image.fromarray(cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2RGB))

# Torch transform pipeline
tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

vecs, valid_ids = [], []

for i, path in paths:
    if not os.path.exists(path):
        print(f"⚠️  Skipping missing image: {path}")
        continue
    try:
        img = clahe(Image.open(path).convert("RGB"))
        with torch.no_grad():
            v = extract(tf(img).unsqueeze(0)).squeeze().numpy()
        vecs.append(v / np.linalg.norm(v))
        valid_ids.append(i)
    except Exception as e:
        print(f"❌ Failed on {path}: {e}")
        continue

# Save valid features
out = np.stack(vecs).astype("float32")
np.save("data/feats_norm.npy", out)

# Save matching index map
pd.DataFrame({"bottle_id": valid_ids}).to_csv("data/id_map_resnet.csv", index=False)

print(f"✅ Saved {len(out)} features to data/feats_norm.npy")
print(f"ℹ️  Index map: data/id_map_resnet.csv")

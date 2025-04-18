#!/usr/bin/env python3
"""
Build two 512‑d CLIP indices over your 500 catalog bottles:
  • feats_clip.npy   ← image embeddings (YOLO‑cropped)
  • text_feats.npy   ← text embeddings of bottle names
Also write id_map.csv with [bottle_id, img_path].
"""
import os, logging
import numpy as np
import pandas as pd
from PIL import Image
import torch, clip
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

device = "cuda" if torch.cuda.is_available() else "cpu"
detector = YOLO("yolov8n.pt")
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

# 1) Load catalog metadata
df = pd.read_csv("data/bottles.csv", dtype={"id": str})
feats_img, ids, img_paths = [], [], []

# 2) Compute image embeddings
for _, row in df.iterrows():
    bid  = row["id"]
    path = f"data/images/{bid}.jpg"
    if not os.path.isfile(path):
        logging.warning(f"{bid}: missing image, skipping")
        continue

    # YOLOv8 crop
    img = Image.open(path).convert("RGB")
    det = detector(path, verbose=False)[0]
    if det.boxes.xyxy.shape[0]:
        boxes = det.boxes.xyxy.cpu().numpy()
        # largest box crop
        areas = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
        x1,y1,x2,y2 = boxes[np.argmax(areas)]
        img = img.crop((x1,y1,x2,y2))

    # CLIP encode + normalize
    with torch.no_grad():
        v = clip_model.encode_image(preprocess(img).unsqueeze(0).to(device))
    v = v / v.norm(dim=-1, keepdim=True)

    feats_img.append(v.cpu().numpy()[0])
    ids.append(bid)
    img_paths.append(path)
    logging.info(f"{bid}: image encoded")

# 3) Save image feats + id_map
feats_img = np.stack(feats_img).astype("float32")
np.save("data/feats_clip.npy", feats_img)
id_map_df = pd.DataFrame({
    "bottle_id": ids,
    "img_path":  img_paths
})
id_map_df.to_csv("data/id_map.csv", index=False)
logging.info("Saved feats_clip.npy + id_map.csv")

# 4) Compute text embeddings over the **same** ids
names = [df.loc[df["id"] == bid, "name"].values[0] for bid in ids]
text_tokens = clip.tokenize(names).to(device)
with torch.no_grad():
    text_embs = clip_model.encode_text(text_tokens)
text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
np.save("data/text_feats.npy", text_embs.cpu().numpy().astype("float32"))
logging.info("Saved text_feats.npy")

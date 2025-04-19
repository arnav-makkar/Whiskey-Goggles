#!/usr/bin/env python3
"""
Build two 512‑d CLIP indices over your 500 catalog bottles:
  • data/feats_img.npy   ← image embeddings (YOLO‑cropped + enhanced)
  • data/feats_txt.npy   ← text embeddings of “name + size + abv + proof”
Also writes data/id_map.csv with [bottle_id, img_path].
"""
import os, logging
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageEnhance
import torch, clip
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

device       = "cuda" if torch.cuda.is_available() else "cpu"
# use the medium YOLOv8s model for more accurate bottle detection
detector     = YOLO("yolov8s.pt")
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

# find the COCO class index for "bottle"
bottle_cls  = next(i for i,n in detector.model.names.items() if n.lower()=="bottle")

# 1) Load catalog metadata
df = pd.read_csv("data/bottles.csv", dtype={"id": str})
ids, img_paths, feats_img = [], [], []

# 2) Encode each bottle image
for _, row in df.iterrows():
    bid  = row["id"]
    path = f"data/images/{bid}.jpg"
    if not os.path.isfile(path):
        logging.warning(f"{bid}: missing image, skipping")
        continue

    # YOLO crop: filter detections to class "bottle", pick largest
    img_pil = Image.open(path).convert("RGB")
    res     = detector(path, verbose=False)[0]
    boxes   = res.boxes.xyxy.cpu().numpy()
    classes = res.boxes.cls.cpu().numpy().astype(int)
    # select only "bottle" class
    bottle_boxes = boxes[classes == bottle_cls]
    if len(bottle_boxes):
        areas = (bottle_boxes[:,2]-bottle_boxes[:,0]) * (bottle_boxes[:,3]-bottle_boxes[:,1])
        x1,y1,x2,y2 = bottle_boxes[np.argmax(areas)]
        img_pil = img_pil.crop((x1, y1, x2, y2))

    # image enhancement: sharpen + boost contrast
    img_pil = img_pil.filter(ImageFilter.SHARPEN)
    img_pil = ImageEnhance.Contrast(img_pil).enhance(1.2)

    # CLIP encode + L2 normalize
    with torch.no_grad():
        v = clip_model.encode_image(preprocess(img_pil).unsqueeze(0).to(device))
    v = v / v.norm(dim=-1, keepdim=True)

    feats_img.append(v.cpu().numpy()[0])
    ids.append(bid)
    img_paths.append(path)
    logging.info(f"{bid}: image encoded")

# 3) Save image feats + id_map
feats_img = np.stack(feats_img).astype("float32")
os.makedirs("data", exist_ok=True)
np.save("data/feats_img.npy", feats_img)
pd.DataFrame({"bottle_id": ids, "img_path": img_paths}) \
  .to_csv("data/id_map.csv", index=False)
logging.info("Saved data/feats_img.npy and data/id_map.csv")

# 4) Encode catalog names with enhanced text prompts
texts = []
for bid in ids:
    rec = df[df["id"]==bid].iloc[0]
    parts = [rec["name"]]
    if not pd.isna(rec.get("size")): parts.append(f"{int(rec['size'])}ml")
    if not pd.isna(rec.get("abv")):  parts.append(f"{rec['abv']}% ABV")
    if not pd.isna(rec.get("proof")): parts.append(f"{int(rec['proof'])}° proof")
    texts.append(" ".join(parts))

tokens = clip.tokenize(texts).to(device)
with torch.no_grad():
    feats_txt = clip_model.encode_text(tokens)
feats_txt = feats_txt / feats_txt.norm(dim=-1, keepdim=True)

np.save("data/feats_txt.npy", feats_txt.cpu().numpy().astype("float32"))
logging.info("Saved data/feats_txt.npy")

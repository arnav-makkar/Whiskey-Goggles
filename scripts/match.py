#!/usr/bin/env python3
"""
Fast & accurate bottle matcher (YOLOv8s + PaddleOCR).

Flow
----
1. Detect bottle with YOLOv8s, crop the largest box
2. Compute CLIP image embedding, compare to catalog
   • if cos_img ≥ 0.98  → exact match, early return
3. Otherwise run PaddleOCR
   • strong OCR (fuzzy ≥ 0.80): 0.8·text + 0.15·fuzzy + 0.05·image
   • weak / no OCR      : 0.7·img→text + 0.3·image
"""
import re, sys, os, zipfile, numpy as np, pandas as pd
from PIL import Image
import torch, clip, cv2
from ultralytics import YOLO
from rapidfuzz import fuzz
from functools import lru_cache

# ─────────────────── CONFIG ────────────────────
TOP_K       = 4
COS_EXACT   = 0.98
STRONG_FUZZ = 0.80
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
STOPWORDS   = {"shutterstock", "image", "stock"}
OCR_DIR     = os.path.expanduser("~/.paddleocr/whl")

# ─────────── Ensure Paddle models exist ─────────
if not os.path.exists(OCR_DIR):
    os.makedirs(os.path.dirname(OCR_DIR), exist_ok=True)
    with zipfile.ZipFile("paddle_models.zip") as zf:
        zf.extractall(os.path.expanduser("~/.paddleocr"))

# ───────────────── Load heavy models ────────────
yolo = YOLO("yolov8s.pt")
clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)
clip_model.eval()

# OCR loader (cached once per Streamlit session)
def _load_ocr():
    from paddleocr import PaddleOCR
    return PaddleOCR(lang="en",
                     use_angle_cls=True,
                     ocr_version="PP-OCRv4",
                     show_log=False,
                     use_gpu=False,
                     det_limit_side_len=640,
                     rec_batch_num=1,
                     cls_batch_num=1)

try:
    import streamlit as st
    ocr_engine = st.cache_resource(_load_ocr)()
except ImportError:
    ocr_engine = _load_ocr()

# ──────────────── Catalog data ──────────────────
feats_img = np.load("data/feats_img.npy")
feats_txt = np.load("data/feats_txt.npy")
id_map    = pd.read_csv("data/id_map.csv")
meta      = pd.read_csv("data/bottles.csv", index_col="id")
names_lo  = [meta.loc[int(b), "name"].lower() for b in id_map["bottle_id"]]

# ────────────────── Helpers ─────────────────────
def _clean(txt: str) -> str:
    txt = txt.lower()
    for w in STOPWORDS:
        txt = txt.replace(w, " ")
    return re.sub(r"[^a-z0-9 ]+", " ", txt).strip()

def _ocr_text(pil_img: Image.Image) -> str:
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    res = ocr_engine.ocr(img_bgr, cls=True)
    text = " ".join([line[1][0] for line in res[0]]) if res and res[0] else ""
    return _clean(text)

def _crop_and_embed(path: str):
    """NO caching here – we want fresh embedding each time."""
    img = Image.open(path).convert("RGB")
    det = yolo(path, verbose=False)[0]
    if det.boxes.xyxy.shape[0]:
        boxes  = det.boxes.xyxy.cpu().numpy()
        area   = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
        x1,y1,x2,y2 = boxes[area.argmax()]
        crop = img.crop((x1, y1, x2, y2))
    else:
        crop = img

    with torch.no_grad():
        vec = clip_model.encode_image(
            preprocess(crop).unsqueeze(0).to(DEVICE)
        )
    vec = vec / vec.norm(dim=-1, keepdim=True)
    return crop, vec.cpu().numpy()[0]

@lru_cache(maxsize=128)
def _embed_text(txt: str) -> np.ndarray:
    with torch.no_grad():
        v = clip_model.encode_text(clip.tokenize([txt]).to(DEVICE))
    return (v / v.norm(dim=-1, keepdim=True)).cpu().numpy()[0]

# ────────────────── Main API ────────────────────
def match(path: str):
    crop, q_img = _crop_and_embed(path)
    cos_img     = feats_img @ q_img
    best_i      = int(cos_img.argmax())

    # exact‑photo?
    if cos_img[best_i] >= COS_EXACT:
        bid = int(id_map.iloc[best_i]["bottle_id"])
        top = meta.loc[bid].to_dict() | {
            "confidence": float(cos_img[best_i]),
            "ref_img":    id_map.iloc[best_i]["img_path"],
            "rank":       1
        }
        alt=[]
        for rk,i in enumerate(np.argsort(-cos_img)[1:TOP_K], start=2):
            bid_i = int(id_map.iloc[i]["bottle_id"])
            alt.append(meta.loc[bid_i].to_dict() | {
                "confidence": float(cos_img[i]),
                "ref_img":    id_map.iloc[i]["img_path"],
                "rank": rk
            })
        return {"status":"in_dataset","top":top,"alt":alt}

    # OCR + hybrid scoring
    ocr_txt = _ocr_text(crop)
    print(f"[DEBUG] OCR text: '{ocr_txt}'", file=sys.stderr)

    if ocr_txt:
        q_txt   = _embed_text(ocr_txt)
        cos_txt = feats_txt @ q_txt
        fuzzy   = np.array(
            [fuzz.partial_ratio(ocr_txt, n)/100.0 for n in names_lo],
            dtype="float32"
        )
        if fuzzy.max() >= STRONG_FUZZ:
            combined = 0.80*cos_txt + 0.15*fuzzy + 0.05*cos_img
        else:
            combined = 0.70*cos_txt + 0.10*fuzzy + 0.20*cos_img
    else:
        cos_i2t  = feats_txt @ q_img
        combined = 0.70*cos_i2t + 0.30*cos_img

    idxs = np.argsort(-combined)[:TOP_K]
    res  = []
    for rk,i in enumerate(idxs, start=1):
        bid = int(id_map.iloc[i]["bottle_id"])
        res.append(meta.loc[bid].to_dict() | {
            "confidence": float(combined[i]),
            "ref_img":    id_map.iloc[i]["img_path"],
            "rank": rk
        })

    status = "unknown" if ocr_txt else "no_text_detected"
    return {"status":status, "top":res[0], "alt":res[1:]}



# ───────────── CLI helper ───────────────
if __name__ == "__main__":
    import pprint
    pprint.pp(match(sys.argv[1]))
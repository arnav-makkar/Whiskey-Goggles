"""
Unified bottle‑matcher

A)  ResNet‑2048  → exact catalog hit  (cos ≥ 0.92)   ← fast & precise
B)  YOLOv8s + CLIP + PaddleOCR                       ← robust fallback

Returns dict:
{
  status : "in_dataset" | "unknown" | "no_text_detected",
  top    : {...},        # best match
  alt    : [...]         # up to 3 alternates
}
"""
from __future__ import annotations
import os, re, sys, zipfile, warnings
from functools import lru_cache

import cv2, numpy as np, pandas as pd
import torch, clip
from PIL import Image
from rapidfuzz import fuzz
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from ultralytics import YOLO

# ───────────────────────────── CONFIG ─────────────────────────────
TOP_K            = 4
RESNET_THRESH    = 0.92       # catalog‑exact threshold
COS_EXACT_CLIP   = 0.98       # for CLIP exact photo (rare fallback)
STRONG_FUZZ      = 0.80
DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
STOPWORDS        = {"shutterstock", "image", "stock"}
OCR_DIR          = os.path.expanduser("~/.paddleocr/whl")

# ───────── ensure Paddle models extracted once ────────────────────
if not os.path.exists(OCR_DIR):
    os.makedirs(os.path.dirname(OCR_DIR), exist_ok=True)
    with zipfile.ZipFile("paddle_models.zip") as zf:
        zf.extractall(os.path.expanduser("~/.paddleocr"))

# ──────────  A)  ResNet‑2048 catalog pipeline  ────────────────────
RES_MODEL  = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
RES_MODEL.eval()
RES_EXTRACT = torch.nn.Sequential(*list(RES_MODEL.children())[:-1])

_tf_res = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

def _clahe_rgb(pil: Image.Image) -> Image.Image:
    arr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    cl = cv2.createCLAHE(2.0,(8,8)).apply(l)
    rgb = cv2.cvtColor(cv2.merge((cl,a,b)), cv2.COLOR_LAB2RGB)
    return Image.fromarray(rgb)

def _resnet_embed(path:str) -> np.ndarray:
    img = _clahe_rgb(Image.open(path).convert("RGB"))
    x   = _tf_res(img).unsqueeze(0)
    with torch.no_grad():
        v = RES_EXTRACT(x).squeeze().numpy()
    v /= np.linalg.norm(v)+1e-8
    return v.astype("float32")

FEATS_NORM = np.load("data/feats_norm.npy")          # (N,2048)
# ------------------------------------------------------------------

# ──────────  B)  YOLO + CLIP + OCR fallback  ──────────────────────
YOLO_DET        = YOLO("yolov8s.pt")
CLIP_MODEL, _tf = clip.load("ViT-B/32", device=DEVICE)
CLIP_MODEL.eval()

def _build_ocr():
    from paddleocr import PaddleOCR
    return PaddleOCR(lang="en",
                     use_angle_cls=True,
                     ocr_version="PP-OCRv4",
                     show_log=False,
                     use_gpu=False,
                     enable_mkldnn=False,
                     det_limit_side_len=640,
                     rec_batch_num=1,
                     cls_batch_num=1)
try:
    import streamlit as st
    OCR_ENGINE = st.cache_resource(_build_ocr)()
except ImportError:
    OCR_ENGINE = _build_ocr()

# catalog data shared
ID_MAP  = pd.read_csv("data/id_map.csv")
META    = pd.read_csv("data/bottles.csv", index_col="id")
NAMES_LO = [META.loc[int(b),"name"].lower() for b in ID_MAP["bottle_id"]]
FEATS_IMG = np.load("data/feats_img.npy")
FEATS_TXT = np.load("data/feats_txt.npy")

# ——— helpers ———
def _clean(t:str)->str:
    t=t.lower()
    for w in STOPWORDS: t=t.replace(w," ")
    return re.sub(r"[^a-z0-9 ]+"," ",t).strip()

def _safe_ocr(bgr):
    global OCR_ENGINE            # ← move this to the very top
    try:
        return OCR_ENGINE.ocr(bgr, cls=True)
    except Exception as e:
        warnings.warn(f"OCR crash, rebuilding… ({e})")
        OCR_ENGINE = _build_ocr()           # rebuild
        return OCR_ENGINE.ocr(bgr, cls=True)


def _ocr_text(pil:Image.Image)->str:
    bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    res = _safe_ocr(bgr)
    txt = " ".join([r[1][0] for r in res[0]]) if res and res[0] else ""
    return _clean(txt)

def _clip_crop_embed(path:str):
    img = Image.open(path).convert("RGB")
    det = YOLO_DET(path,verbose=False)[0]
    if det.boxes.xyxy.shape[0]:
        xyxy = det.boxes.xyxy.cpu().numpy()
        areas = (xyxy[:,2] - xyxy[:,0]) * (xyxy[:,3] - xyxy[:,1])
        largest_idx = int(areas.argmax())
        x1, y1, x2, y2 = xyxy[largest_idx]
        img = img.crop((x1,y1,x2,y2))
    with torch.no_grad():
        v = CLIP_MODEL.encode_image(_tf(img).unsqueeze(0).to(DEVICE))
    v = v/ v.norm(dim=-1,keepdim=True)
    return img, v.cpu().numpy()[0]

@lru_cache(maxsize=128)
def _embed_text(txt:str)->np.ndarray:
    with torch.no_grad():
        v = CLIP_MODEL.encode_text(clip.tokenize([txt]).to(DEVICE))
    return (v/v.norm(dim=-1,keepdim=True)).cpu().numpy()[0]

# ─────────────────────  MAIN  ──────────────────────
def match(path:str):
    # ---------- Stage A : ResNet exact ----------
    q_res   = _resnet_embed(path)
    sims_r  = FEATS_NORM @ q_res          # cosine already normalised
    best_r  = int(sims_r.argmax())
    if sims_r[best_r] >= RESNET_THRESH:
        bid = int(ID_MAP.iloc[best_r]["bottle_id"])
        top = META.loc[bid].to_dict() | {
            "confidence": float(sims_r[best_r]),
            "ref_img":    ID_MAP.iloc[best_r]["img_path"],
            "rank":1
        }
        alt=[]
        for rk,i in enumerate(np.argsort(-sims_r)[1:TOP_K], start=2):
            b2 = int(ID_MAP.iloc[i]["bottle_id"])
            alt.append(META.loc[b2].to_dict() | {
                "confidence": float(sims_r[i]),
                "ref_img":    ID_MAP.iloc[i]["img_path"],
                "rank": rk
            })
        return {"status":"in_dataset","top":top,"alt":alt}

    # ---------- Stage B : Hybrid fallback ----------
    crop, q_img = _clip_crop_embed(path)
    sims_c = FEATS_IMG @ q_img
    best_c = int(sims_c.argmax())
    if sims_c[best_c] >= COS_EXACT_CLIP:
        bid = int(ID_MAP.iloc[best_c]["bottle_id"])
        return {
            "status":"in_dataset",
            "top": META.loc[bid].to_dict() | {
                "confidence": float(sims_c[best_c]),
                "ref_img": ID_MAP.iloc[best_c]["img_path"],
                "rank":1},
            "alt": []
        }

    ocr_txt = _ocr_text(crop)
    if ocr_txt:
        cos_txt = FEATS_TXT @ _embed_text(ocr_txt)
        fuzzy   = np.array([fuzz.partial_ratio(ocr_txt,n)/100 for n in NAMES_LO],
                           dtype="float32")
        if fuzzy.max() >= STRONG_FUZZ:
            combined = 0.80*cos_txt + 0.15*fuzzy + 0.05*sims_c
        else:
            combined = 0.70*cos_txt + 0.10*fuzzy + 0.20*sims_c
    else:
        cos_i2t  = FEATS_TXT @ q_img
        combined = 0.70*cos_i2t + 0.30*sims_c

    idxs = np.argsort(-combined)[:TOP_K]
    out=[]
    for rk,i in enumerate(idxs,1):
        b = int(ID_MAP.iloc[i]["bottle_id"])
        out.append(META.loc[b].to_dict() | {
            "confidence": float(combined[i]),
            "ref_img": ID_MAP.iloc[i]["img_path"],
            "rank": rk
        })

    status = "unknown" if ocr_txt else "no_text_detected"
    return {"status":status,"top":out[0],"alt":out[1:]}

# ───────────── CLI helper ───────────────
if __name__ == "__main__":
    import pprint
    pprint.pp(match(sys.argv[1]))

#!/usr/bin/env python3
"""
Hybrid CLIP matcher
 • Exact catalog photo → confidence 0.9999
 • Unknown bottle:
        – strong OCR  → 60 % text + 30 % OCR + 10 % image
        – weak/empty  → 70 % image + 25 % text + 5 % OCR
"""
import re, subprocess, tempfile
import numpy as np, pandas as pd
from PIL import Image
import torch, clip
from ultralytics import YOLO
from rapidfuzz import fuzz

# ---------- config ----------
TOP_K      = 4
COS_EXACT  = 0.98    # exact‑photo threshold
FUZZ_TH    = 70      # "strong OCR" fuzzy %
device     = "cuda" if torch.cuda.is_available() else "cpu"
STOPWORDS  = {"shutterstock", "image", "stock"}
# -----------------------------

# models & indices
detector      = YOLO("yolov8n.pt")
clip_model, preprocess = clip.load("ViT-B/32", device=device); clip_model.eval()

feats_img     = np.load("data/feats_clip.npy")          # (N,512)
feats_txt     = np.load("data/text_feats.npy")          # (N,512)
id_map        = pd.read_csv("data/id_map.csv")
meta          = pd.read_csv("data/bottles.csv", index_col="id")
names_lower   = [meta.loc[int(b),"name"].lower() for b in id_map["bottle_id"]]

# ---------- helpers ----------
def _clean(text:str)->str:
    text = text.lower()
    for w in STOPWORDS: text = text.replace(w, " ")
    return re.sub(r"[^a-z0-9 ]+", " ", text).strip()

def _ocr_on_pil(pil_img:Image.Image)->str:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
        pil_img.save(tmp.name)
        try:
            out = subprocess.run(
                ["tesseract", tmp.name, "stdout"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                check=True, text=True
            ).stdout
            return _clean(out)
        except: return ""

def _embed_image(path:str):
    img = Image.open(path).convert("RGB")
    # YOLO crop
    det = detector(path, verbose=False)[0]
    if det.boxes.xyxy.shape[0]:
        boxes = det.boxes.xyxy.cpu().numpy()
        areas = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
        x1,y1,x2,y2 = boxes[np.argmax(areas)]
        label_crop  = img.crop((x1,y1,x2,y2))
    else:
        label_crop = img

    # OCR on the crop (better text read)
    ocr_txt = _ocr_on_pil(label_crop)

    with torch.no_grad():
        vec = clip_model.encode_image(preprocess(label_crop).unsqueeze(0).to(device))
    vec = vec / vec.norm(dim=-1, keepdim=True)
    return vec.cpu().numpy()[0], ocr_txt

def _embed_text(txt:str):
    with torch.no_grad():
        t = clip_model.encode_text(clip.tokenize(txt).to(device))
    return (t / t.norm(dim=-1, keepdim=True)).cpu().numpy()[0]
# --------------------------------


def match(path:str):
    # 1) image vec + OCR
    q_img, ocr_txt = _embed_image(path)
    cos_img = feats_img @ q_img          # (N,)

    # exact?
    best_i = np.argmax(cos_img)
    if cos_img[best_i] >= COS_EXACT:
        bid = int(id_map.iloc[best_i]["bottle_id"])
        exact = meta.loc[bid].to_dict() | {
            "confidence": 0.9999,
            "ref_img":    id_map.iloc[best_i]["img_path"],
            "rank":       1
        }
        # prepare alternates
        alt = []
        for rk,i in enumerate(np.argsort(-cos_img)[1:TOP_K],2):
            bid_i = int(id_map.iloc[i]["bottle_id"])
            alt.append(meta.loc[bid_i].to_dict() | {
                "confidence": float(cos_img[i]),
                "ref_img": id_map.iloc[i]["img_path"],
                "rank": rk
            })
        return {"status":"in_dataset","top":exact,"alt":alt}

    # 2) unknown → text first
    q_txt  = _embed_text(ocr_txt if ocr_txt else "unknown bottle")
    cos_txt = feats_txt @ q_txt

    fuzzy = np.array([
        fuzz.partial_ratio(ocr_txt, nm)/100.0 if ocr_txt else 0.0
        for nm in names_lower
    ], dtype="float32")

    strong = fuzzy >= (FUZZ_TH/100)

    # combined = np.where(
    #     strong,
    #     0.5*cos_txt + 0.4*fuzzy + 0.1*cos_img,
    #     0.2*cos_img + 0.5*cos_txt + 0.3*fuzzy
    # )

    # combined = np.where(
    #     strong,
    #     0.9*cos_txt + 0.0*fuzzy + 0.1*cos_img,
    #     0.0*cos_img + 0.9*cos_txt + 0.1*fuzzy
    # )

    # combined = np.where(
    #     strong,
    #     0.6*cos_txt + 0.3*fuzzy + 0.1*cos_img,
    #     0.7*cos_img + 0.25*cos_txt + 0.05*fuzzy
    # )

    # --- new variables -------------
    cos_img2txt = feats_txt @ q_img        # image⇢text similarity (shape N)
    have_ocr    = bool(ocr_txt)            # True if we got any text

    # Re‑compute 'combined'
    if have_ocr and fuzzy.max() >= 0.70:
        # Good OCR → keep text‑heavy path
        combined = 0.6 * cos_txt + 0.3 * fuzzy + 0.1 * cos_img
    else:
        # Weak/empty OCR → blend visual with CLIP image→text
        # 0.5 visual + 0.4 image→text + 0.1 fuzzy (tiny help if any)
        combined = 0.5 * cos_img + 0.5 * cos_img2txt + 0.1 * fuzzy


    idxs = np.argsort(-combined)[:TOP_K]
    res=[]
    for rk,i in enumerate(idxs,1):
        bid = int(id_map.iloc[i]["bottle_id"])
        res.append(meta.loc[bid].to_dict() | {
            "confidence": float(combined[i]),
            "ref_img": id_map.iloc[i]["img_path"],
            "rank": rk
        })

    return {"status":"unknown","top":res[0],"alt":res[1:]}


if __name__ == "__main__":
    import sys, pprint
    pprint.pp(match(sys.argv[1]))
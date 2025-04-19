#!/usr/bin/env python3
"""
Fast & accurate bottle matcher (YOLOv8s + PaddleOCR).

Flow:
1. Crop bottle with YOLOv8s.
2. Compute CLIP image embedding, compare to catalog.
   • if cos_img ≥ 0.98 → exact match, early return.
3. Otherwise run PaddleOCR (angle_cls=True).
   • strong OCR (fuzzy ≥ 0.80): 0.8·text + 0.15·fuzzy + 0.05·image
   • weak/empty OCR        : 0.7·img→text + 0.3·image
"""
import re, sys, numpy as np, pandas as pd
from PIL import Image
import torch, clip, cv2
from ultralytics import YOLO
from rapidfuzz import fuzz
from paddleocr import PaddleOCR

# ---------------- CONFIG ----------------
TOP_K       = 4
COS_EXACT   = 0.98
STRONG_FUZZ = 0.80   # fuzzy ratio threshold
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
STOPWORDS   = {"shutterstock","image","stock"}
# ----------------------------------------

# Models
yolo         = YOLO("yolov8s.pt")                     # better detector
clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)
clip_model.eval()
ocr_engine   = PaddleOCR(lang="en", use_angle_cls=True, use_gpu=False,
                         rec_batch_num=1, cls_batch_num=1,
                         det_limit_side_len=640)

# Catalog embeddings
feats_img = np.load("data/feats_img.npy")
feats_txt = np.load("data/feats_txt.npy")
id_map    = pd.read_csv("data/id_map.csv")
meta      = pd.read_csv("data/bottles.csv", index_col="id")
names_lo  = [meta.loc[int(b),"name"].lower() for b in id_map["bottle_id"]]

# -------------- helpers -----------------
def clean(s:str)->str:
    s = s.lower()
    for w in STOPWORDS: s = s.replace(w," ")
    return re.sub(r"[^a-z0-9 ]+"," ",s).strip()

def ocr_label(pil:Image.Image)->str:
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    res = ocr_engine.ocr(img, cls=True)
    txt = " ".join([ln[1][0] for ln in res[0]]) if res and res[0] else ""
    return clean(txt)

def crop_and_embed(path:str):
    img = Image.open(path).convert("RGB")
    det = yolo(path, verbose=False)[0]
    if len(det.boxes.xyxy):
        boxes = det.boxes.xyxy.cpu().numpy()
        areas = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
        x1,y1,x2,y2 = boxes[np.argmax(areas)]
        crop = img.crop((x1,y1,x2,y2))
    else:
        crop = img

    with torch.no_grad():
        vec = clip_model.encode_image(preprocess(crop).unsqueeze(0).to(DEVICE))
    vec = vec/vec.norm(dim=-1,keepdim=True)
    return crop, vec.cpu().numpy()[0]

def embed_text(txt:str):
    with torch.no_grad():
        t = clip_model.encode_text(clip.tokenize([txt]).to(DEVICE))
    return (t/t.norm(dim=-1,keepdim=True)).cpu().numpy()[0]

# -------------- main match --------------
def match(path:str):
    crop, q_img = crop_and_embed(path)
    cos_img = feats_img @ q_img
    top_i   = int(np.argmax(cos_img))

    # ---- exact match early‑exit ----
    if cos_img[top_i] >= COS_EXACT:
        bid = int(id_map.iloc[top_i]["bottle_id"])
        top = meta.loc[bid].to_dict() | {
            "confidence": float(cos_img[top_i]),
            "ref_img":    id_map.iloc[top_i]["img_path"],
            "rank":1
        }
        alt=[]
        for rk,i in enumerate(np.argsort(-cos_img)[1:TOP_K],2):
            b2 = int(id_map.iloc[i]["bottle_id"])
            alt.append(meta.loc[b2].to_dict() | {
                "confidence": float(cos_img[i]),
                "ref_img":    id_map.iloc[i]["img_path"],
                "rank": rk
            })
        return {"status":"in_dataset", "top":top, "alt":alt}

    # ---- run OCR only now (saves time) ----
    ocr_txt = ocr_label(crop)
    print(f"[DEBUG] OCR: '{ocr_txt}'", file=sys.stderr)

    if ocr_txt:
        q_txt   = embed_text(ocr_txt)
        cos_txt = feats_txt @ q_txt
        fuzzy   = np.array([fuzz.partial_ratio(ocr_txt,n)/100.0 for n in names_lo],dtype="float32")
        fmax    = fuzzy.max()

        if fmax >= STRONG_FUZZ:
            combined = 0.80*cos_txt + 0.15*fuzzy + 0.05*cos_img
        else:
            combined = 0.70*cos_txt + 0.10*fuzzy + 0.20*cos_img
    else:
        # no OCR text → image→text fallback
        cos_i2t  = feats_txt @ q_img
        combined = 0.70*cos_i2t + 0.30*cos_img

    idxs = np.argsort(-combined)[:TOP_K]
    results=[]
    for rk,i in enumerate(idxs,1):
        bid = int(id_map.iloc[i]["bottle_id"])
        results.append(meta.loc[bid].to_dict() | {
            "confidence": float(combined[i]),
            "ref_img":    id_map.iloc[i]["img_path"],
            "rank": rk
        })

    status = "unknown"
    if not ocr_txt: status = "no_text_detected"

    return {"status":status, "top":results[0], "alt":results[1:]}

# -------------- CLI ---------------------
if __name__ == "__main__":
    import pprint
    pprint.pp(match(sys.argv[1]))

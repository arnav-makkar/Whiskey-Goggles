# gradio_app.py
import os, sys, cv2, torch, clip, numpy as np, pandas as pd, gradio as gr
from PIL import Image
from ultralytics import YOLO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "scripts")))
from scripts.match import match          # <-- your existing matcher

# ────────── GLOBAL MODELS ──────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL, PREPROCESS = clip.load("ViT-B/32", device=DEVICE); CLIP_MODEL.eval()

DETECTOR  = YOLO("yolov8s.pt")
BOTTLE_CLS = next(k for k,v in DETECTOR.model.names.items() if v.lower()=="bottle")

FEATS_IMG = np.load("data/feats_img.npy")
ID_MAP    = pd.read_csv("data/id_map.csv")
ID_LIST   = ID_MAP["bottle_id"].astype(int).tolist()
META_DF   = pd.read_csv("data/bottles.csv", index_col="id")

# ────────── HELPERS ──────────
def embed_pil(pil: Image.Image) -> np.ndarray:
    with torch.no_grad():
        vec = CLIP_MODEL.encode_image(PREPROCESS(pil).unsqueeze(0).to(DEVICE))
        return (vec / vec.norm(dim=-1, keepdim=True)).cpu().numpy()[0]

def annotate_frame(frame: np.ndarray) -> np.ndarray:
    """YOLO detect, CLIP match, draw boxes on a BGR frame."""
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = DETECTOR.predict(img_rgb, verbose=False)[0]
    out = frame.copy()

    for xyxy, cls in zip(res.boxes.xyxy.cpu().numpy(),
                         res.boxes.cls.cpu().numpy().astype(int)):
        if cls != BOTTLE_CLS:
            continue
        x1,y1,x2,y2 = map(int, xyxy)
        crop = img_rgb[y1:y2, x1:x2]
        if crop.size == 0: 
            continue
        vec = embed_pil(Image.fromarray(crop))
        sims = FEATS_IMG @ vec
        best = int(np.argmax(sims))
        bid  = ID_LIST[best]
        name = META_DF.loc[bid, "name"]
        msrp = META_DF.loc[bid, "avg_msrp"]
        shelf= META_DF.loc[bid, "shelf_price"]
        conf = sims[best]

        cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
        lbl1 = f"{name[:22]}… {conf*100:.1f}%"
        lbl2 = f"MSRP ${msrp:.0f} | Shelf ${shelf:.0f}"
        ytxt = y1-10 if y1>25 else y2+20
        cv2.putText(out, lbl1, (x1, ytxt),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
        cv2.putText(out, lbl2, (x1, ytxt+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
    return out

def run_match(pil_img: Image.Image):
    """Wrapper used by Upload / Take‑Photo buttons."""
    tmp = "temp_query.jpg"
    pil_img.save(tmp)
    res = match(tmp)
    top = res["top"]
    md = (
        f"### {top['name']}\n"
        f"**Confidence:** {top['confidence']*100:.2f}%\n\n"
        f"**MSRP:** ${top.get('avg_msrp','N/A')} | "
        f"**Shelf:** ${top.get('shelf_price','N/A')}"
    )
    return pil_img, md, top["ref_img"]

# ────────── GRADIO UI ──────────
with gr.Blocks(title="Whisky Goggles") as demo:
    gr.Markdown("## 🥃 Whisky Goggles – CLIP + YOLO + OCR")

    with gr.Tab("📁 Upload Photo"):
        upload_img = gr.Image(type="pil", label="Upload a bottle photo")
        btn_u      = gr.Button("🔍 Match")
        out_img_u  = gr.Image(label="Query")
        out_md_u   = gr.Markdown()
        out_ref_u  = gr.Image(label="Catalog Reference")
        btn_u.click(run_match, upload_img, [out_img_u, out_md_u, out_ref_u])

    with gr.Tab("📸 Take Photo"):
        cam_img = gr.Image(sources="webcam", type="pil", label="Camera")
        btn_c   = gr.Button("🔍 Match")
        out_img_c = gr.Image(label="Captured")
        out_md_c  = gr.Markdown()
        out_ref_c = gr.Image(label="Catalog Reference")
        btn_c.click(run_match, cam_img, [out_img_c, out_md_c, out_ref_c])

    with gr.Tab("🎥 Real Time"):
        gr.Markdown("Move a bottle in front of your webcam:")
        live_src  = gr.Image(sources="webcam", streaming=True, type="numpy", label="Webcam Feed")
        live_out  = gr.Image(label="Live Detection")
        # stream frames through annotate_frame
        live_src.stream(fn=annotate_frame, outputs=live_out)

demo.launch()

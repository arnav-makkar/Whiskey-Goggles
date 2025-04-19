# app/streamlit_app.py
import streamlit as st
import numpy as np, cv2, torch
import clip
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# —── GLOBAL SETUP ──────────────────────────────────────────────────────────────
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP
CLIP_MODEL, PREPROCESS = clip.load("ViT-B/32", device=DEVICE)
CLIP_MODEL.eval()

# Load YOLOv8n for bottle detection
DETECTOR    = YOLO("yolov8n.pt")

# Load catalog embeddings + metadata
FEATS_IMG   = np.load("data/feats_img.npy")        # shape (N,512)
ID_MAP_DF   = __import__("pandas").read_csv("data/id_map.csv")
ID_LIST     = ID_MAP_DF["bottle_id"].astype(str).tolist()
META_DF     = __import__("pandas").read_csv("data/bottles.csv", index_col="id")

# Embedding helper
def embed_pil(img: Image.Image) -> np.ndarray:
    x = PREPROCESS(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        v = CLIP_MODEL.encode_image(x)
        v = v / v.norm(dim=-1, keepdim=True)
    return v.cpu().numpy()[0]

# Core detect+annotate on a BGR frame
def annotate_frame(frame_bgr: np.ndarray) -> np.ndarray:
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = DETECTOR.predict(img_rgb, verbose=False)[0]
    out = frame_bgr.copy()

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        crop = img_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        pil_crop = Image.fromarray(crop)
        vec = embed_pil(pil_crop)
        sims = FEATS_IMG @ vec
        best = int(np.argmax(sims))
        bid  = int(ID_LIST[best])
        name = META_DF.loc[bid, "name"]
        msrp = META_DF.loc[bid, "avg_msrp"]
        shelf= META_DF.loc[bid, "shelf_price"]
        conf = sims[best]

        # draw
        cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
        label = f"{name[:20]}… {conf:.2f}"
        info  = f"MSRP${msrp:.0f}/Shelf${shelf:.0f}"
        ytxt = y1-10 if y1>20 else y2+20
        cv2.putText(out, label, (x1, ytxt),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        cv2.putText(out, info,  (x1, ytxt+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

    return out

# VideoTransformer for real-time
class BottleTransformer(VideoTransformerBase):
    def transform(self, frame):
        img_bgr = frame.to_ndarray(format="bgr24")
        annotated = annotate_frame(img_bgr)
        return annotated

# —── STREAMLIT UI ─────────────────────────────────────────────────────────────
st.set_page_config(layout="wide")
st.title("🥃 Whisky Goggles – 3‑Mode Demo")

tabs = st.tabs(["Upload Photo","Take Photo","Real Time"])

# — Upload Photo Tab —
with tabs[0]:
    st.header("Upload a photo")
    uploaded = st.file_uploader("", type=["jpg","png","jpeg"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Input Image", use_column_width=True)
        # annotate
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        ann = annotate_frame(bgr)
        st.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), caption="Detected", use_column_width=True)

# — Take Photo Tab —
with tabs[1]:
    st.header("Take a photo with your camera")
    cam_img = st.camera_input("Snap here")
    if cam_img:
        img = Image.open(cam_img).convert("RGB")
        st.image(img, caption="Input Photo", use_column_width=True)
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        ann = annotate_frame(bgr)
        st.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), caption="Detected", use_column_width=True)

# — Real Time Tab —
with tabs[2]:
    st.header("Live camera (press ‘q’ in window to quit)")
    webrtc_streamer(
        key="live",
        mode=WebRtcMode.SENDRECV,
        video_transformer_factory=BottleTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# app/streamlit_app.py
"""
Streamlit‑based Whisky Goggles demo

• Tab 1  Upload Photo   → runs scripts.match → shows best match + alternates
• Tab 2  Take Photo     → same logic, camera input
• Tab 3  Real‑Time      → live multi‑bottle detection via WebRTC
      – detects ONLY objects whose YOLO class == 'bottle'
      – draws green box + name + MSRP / Shelf price
"""
import os, sys, cv2, numpy as np, torch, streamlit as st
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.match import match

import clip
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import pandas as pd

# ───────────────────────── GLOBAL MODELS ──────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# CLIP for real‑time embedding
CLIP_MODEL, PREPROCESS = clip.load("ViT-B/32", device=DEVICE)
CLIP_MODEL.eval()

# YOLOv8s – higher accuracy, we’ll filter class “bottle”
DETECTOR  = YOLO("yolov8s.pt")
BOTTLE_CLS = next(k for k,v in DETECTOR.model.names.items() if v.lower()=="bottle")

# Catalog embeddings + metadata
FEATS_IMG = np.load("data/feats_img.npy")          # (N,512)
ID_MAP_DF = pd.read_csv("data/id_map.csv")
ID_LIST   = ID_MAP_DF["bottle_id"].astype(int).tolist()
META_DF   = pd.read_csv("data/bottles.csv", index_col="id")

def embed_pil(pil: Image.Image) -> np.ndarray:
    with torch.no_grad():
        vec = CLIP_MODEL.encode_image(PREPROCESS(pil).unsqueeze(0).to(DEVICE))
    return (vec / vec.norm(dim=-1, keepdim=True)).cpu().numpy()[0]

# Core detection + annotation (BGR frame)
def annotate_frame(frame_bgr: np.ndarray) -> np.ndarray:
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = DETECTOR.predict(img_rgb, verbose=False)[0]
    out = frame_bgr.copy()

    for xyxy, cls in zip(res.boxes.xyxy.cpu().numpy(),
                         res.boxes.cls.cpu().numpy().astype(int)):
        if cls != BOTTLE_CLS:        # ignore non‑bottles
            continue
        x1,y1,x2,y2 = map(int, xyxy)
        crop = img_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        pil_crop = Image.fromarray(crop)
        vec      = embed_pil(pil_crop)
        sims     = FEATS_IMG @ vec
        best_idx = int(np.argmax(sims))
        bid      = ID_LIST[best_idx]
        name     = META_DF.loc[bid, "name"]
        msrp     = META_DF.loc[bid, "avg_msrp"]
        shelf    = META_DF.loc[bid, "shelf_price"]
        conf     = sims[best_idx]

        cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
        txt1 = f"{name[:22]}… {conf*100:.1f}%"
        txt2 = f"MSRP ${msrp:.0f}  Shelf ${shelf:.0f}"
        y_txt = y1 - 12 if y1 > 25 else y2 + 15
        cv2.putText(out, txt1, (x1, y_txt),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(out, txt2, (x1, y_txt+16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1, cv2.LINE_AA)
    return out

class BottleTransformer(VideoTransformerBase):
    def transform(self, frame):
        img_bgr = frame.to_ndarray(format="bgr24")
        return annotate_frame(img_bgr)

# ─────────────────────────── UI LAYOUT ────────────────────────────
st.set_page_config(page_title="🥃 Whisky Goggles", layout="centered")
st.title("🥃 Whisky Goggles")
st.caption("Bottle identifier using CLIP + YOLO + PaddleOCR (500‑bottle catalog)")
tab_upload, tab_camera, tab_live = st.tabs(["📁 Upload Photo","📸 Take Photo","🎥 Real Time"])

# ---------- Tab 1: Upload ----------
with tab_upload:
    st.subheader("Upload a bottle photo")
    uploaded = st.file_uploader("Choose an image", type=["jpg","jpeg","png"])
    if uploaded:
        tmp = "tmp_upload.jpg"
        with open(tmp, "wb") as f: f.write(uploaded.read())
        st.image(tmp, caption="Query", use_column_width=True)
        with st.spinner("Matching..."):
            res = match(tmp)

        top = res["top"]
        c1, c2 = st.columns([3,2])
        with c1:
            st.markdown(f"### {top['name']}")
            st.markdown(f"**Confidence:** {top['confidence']*100:.2f}%")
            st.markdown(f"MSRP: **${top.get('avg_msrp','N/A')}** &nbsp;&nbsp; Shelf: **${top.get('shelf_price','N/A')}**")
            status = {"in_dataset":"✅ Exact catalog match",
                      "unknown":"⚠️ Closest match (not in catalog)",
                      "no_text_detected":"🔎 No label text detected"}[res["status"]]
            st.info(status)
        with c2:
            st.image(top["ref_img"], caption="Catalog reference")

        with st.expander("Other good matches"):
            for alt in res["alt"]:
                st.markdown(f"{alt['rank']}. **{alt['name']}** ({alt['confidence']*100:.2f}%)")

# ---------- Tab 2: Camera ----------
with tab_camera:
    st.subheader("Take a Photo")
    st.caption("Click ▶️ to open the camera and take a photo of your bottle.")

    if "cam_mode_on" not in st.session_state:
        st.session_state["cam_mode_on"] = False

    col1, col2 = st.columns([1, 3])
    with col1:
        take_photo = st.button("▶️ Take Photo")
    with col2:
        close_photo = st.button("⏹️ Close Camera")

    if take_photo:
        st.session_state["cam_mode_on"] = True
    if close_photo:
        st.session_state["cam_mode_on"] = False

    if st.session_state["cam_mode_on"]:
        cam_img = st.camera_input("📸 Capture your bottle")
        if cam_img:
            tmp = "tmp_camera.jpg"
            with open(tmp, "wb") as f:
                f.write(cam_img.read())
            st.image(tmp, caption="Captured Image", use_column_width=True)
            with st.spinner("Matching..."):
                res = match(tmp)
            top = res["top"]
            st.markdown(f"### {top['name']}  ({top['confidence']*100:.2f}%)")
            st.image(top["ref_img"], caption="Catalog reference")
            with st.expander("Other matches"):
                for alt in res["alt"]:
                    st.markdown(f"{alt['rank']}. {alt['name']} ({alt['confidence']*100:.2f}%)")
    else:
        st.info("Camera is off. Click ▶️ to open the camera.")

# ---------- Tab 3: Real‑Time ----------
with tab_live:
    st.subheader("Live bottle detector (WebRTC)")
    st.caption("Move a bottle in front of your webcam – only objects recognised as bottles will be boxed and labelled.")

    if "camera_on" not in st.session_state:
        st.session_state["camera_on"] = False

    col1, col2 = st.columns([1, 3])
    with col1:
        start = st.button("▶️ Start Camera")
    with col2:
        stop = st.button("⏹️ Stop Camera")

    if start:
        st.session_state["camera_on"] = True
    if stop:
        st.session_state["camera_on"] = False

    if st.session_state["camera_on"]:
        webrtc_streamer(
            key="whisky-live",
            mode=WebRtcMode.SENDRECV,
            video_transformer_factory=BottleTransformer,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    else:
        st.info("Camera is off. Click ▶️ to start real-time detection.")

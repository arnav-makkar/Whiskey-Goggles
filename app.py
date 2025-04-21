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
from PIL import Image

# Must be first Streamlit call
st.set_page_config(page_title="🥃 Whisky Goggles", layout="centered")

# Add project root to path so we can import scripts.match
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.match import match

import clip
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import pandas as pd

# ──────────────── CACHING ────────────────

@st.cache_resource(show_spinner=False)
def load_models():
    """Load CLIP, YOLO once per session."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    yolo = YOLO("yolov8s.pt")
    # find bottle class index
    bottle_cls = next(k for k,v in yolo.model.names.items() if v.lower()=="bottle")
    return clip_model, preprocess, yolo, bottle_cls, device

@st.cache_data(show_spinner=False)
def load_catalog():
    """Load precomputed embeddings and metadata once per session."""
    feats_img = np.load("data/feats_img.npy")
    id_map_df = pd.read_csv("data/id_map.csv")
    id_list   = id_map_df["bottle_id"].astype(int).tolist()
    meta_df   = pd.read_csv("data/bottles.csv", index_col="id")
    return feats_img, id_map_df, id_list, meta_df

CLIP_MODEL, PREPROCESS, DETECTOR, BOTTLE_CLS, DEVICE = load_models()
FEATS_IMG, ID_MAP_DF, ID_LIST, META_DF            = load_catalog()

# ───────────────── HELPERS ─────────────────

def embed_pil(pil: Image.Image) -> np.ndarray:
    with torch.no_grad():
        vec = CLIP_MODEL.encode_image(PREPROCESS(pil).unsqueeze(0).to(DEVICE))
        vec = vec / vec.norm(dim=-1, keepdim=True)
    return vec.cpu().numpy()[0]

def annotate_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """Detect bottles, match via CLIP, and draw on a BGR frame."""
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = DETECTOR.predict(img_rgb, verbose=False)[0]
    out = frame_bgr.copy()

    for xyxy, cls in zip(results.boxes.xyxy.cpu().numpy(),
                         results.boxes.cls.cpu().numpy().astype(int)):
        if cls != BOTTLE_CLS:
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
        rec      = META_DF.loc[bid]
        name, msrp, shelf = rec["name"], rec["avg_msrp"], rec["shelf_price"]
        conf     = sims[best_idx]

        # Draw box + text
        cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
        lbl1 = f"{name[:20]}… {conf*100:.1f}%"
        lbl2 = f"MSRP ${msrp:.0f} | Shelf ${shelf:.0f}"
        y_txt = y1-10 if y1>20 else y2+20
        cv2.putText(out, lbl1, (x1, y_txt),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        cv2.putText(out, lbl2, (x1, y_txt+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
    return out

class BottleTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        return annotate_frame(img)

# ───────────────── UI LAYOUT ─────────────────

st.title("🥃 Whisky Goggles")
st.caption("Bottle identifier using CLIP + YOLO + PaddleOCR (500‑bottle catalog)")

tab_upload, tab_camera, tab_live = st.tabs(
    ["📁 Upload Photo","📸 Take Photo","🎥 Real Time"]
)

# --- Tab 1: Upload Photo ---
with tab_upload:
    st.subheader("Upload a bottle photo")
    uploaded = st.file_uploader("Choose an image", type=["jpg","jpeg","png"])
    if uploaded:
        tmp = "tmp_upload.jpg"
        with open(tmp, "wb") as f: f.write(uploaded.read())
        st.image(tmp, caption="Query", use_container_width=True)
        with st.spinner("Matching..."):
            res = match(tmp)

        top = res["top"]
        c1, c2 = st.columns([3,2])
        with c1:
            st.markdown(f"### {top['name']}")
            st.markdown(f"**Confidence:** {top['confidence']*100:.2f}%")
            

            st.write(f"**MSRP:** ${top.get('avg_msrp','N/A')}  Shelf: ${top.get('shelf_price','N/A')}")



            status = {
                "in_dataset":      "✅ Exact catalog match",
                "unknown":         "⚠️ Closest match (not in catalog)",
                "no_text_detected":"🔎 No label text detected"
            }[res["status"]]
            st.info(status)
        with c2:
            st.image(top["ref_img"], caption="Catalog reference")

        with st.expander("Other good matches"):
            for alt in res["alt"]:
                st.markdown(
                    f"{alt['rank']}. **{alt['name']}** "
                    f"({alt['confidence']*100:.2f}%)"
                )

# --- Tab 2: Take Photo ---
with tab_camera:
    st.subheader("Take a Photo")
    st.caption("Click ▶️ to open the camera and photograph your bottle.")
    if "cam_on" not in st.session_state:
        st.session_state.cam_on = False

    col1, col2 = st.columns([1,3])
    if col1.button("▶️ Take Photo"):
        st.session_state.cam_on = True
    if col2.button("⏹️ Close Camera"):
        st.session_state.cam_on = False

    if st.session_state.cam_on:
        cam_img = st.camera_input("📸 Capture your bottle")
        if cam_img:
            tmp = "tmp_camera.jpg"
            with open(tmp, "wb") as f: f.write(cam_img.read())
            st.image(tmp, caption="Captured", use_container_width=True)
            with st.spinner("Matching..."):
                res = match(tmp)
            top = res["top"]
            st.markdown(f"### {top['name']} ({top['confidence']*100:.2f}%)")
            st.image(top["ref_img"], caption="Catalog reference")
            with st.expander("Other matches"):
                for alt in res["alt"]:
                    st.markdown(
                        f"{alt['rank']}. {alt['name']} "
                        f"({alt['confidence']*100:.2f}%)"
                    )
    else:
        st.info("Camera is off. Click ▶️ to open it.")

# --- Tab 3: Real‑Time ---
with tab_live:
    st.subheader("Live bottle detector (WebRTC)")
    st.caption("Move a bottle in front of your webcam – only bottles are boxed & labelled.")
    if "live_on" not in st.session_state:
        st.session_state.live_on = False

    col1, col2 = st.columns([1,3])
    if col1.button("▶️ Start Camera"):
        st.session_state.live_on = True
    if col2.button("⏹️ Stop Camera"):
        st.session_state.live_on = False

    if st.session_state.live_on:
        webrtc_streamer(
            key="whisky-live",
            mode=WebRtcMode.SENDRECV,
            video_transformer_factory=BottleTransformer,
            media_stream_constraints={"video":True,"audio":False},
            async_processing=True,
        )
    else:
        st.info("Camera is off. Click ▶️ to start real-time detection.")

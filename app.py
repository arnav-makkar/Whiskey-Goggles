# app/streamlit_app.py
"""
Streamlit‑based Whisky Goggles demo

• Tab 1  Upload Photo   → runs scripts.match → shows best match + alternates
• Tab 2  Take Photo     → same logic, camera input
• Tab 3  Real‑Time      → live multi‑bottle detection via WebRTC
      – detects ONLY objects whose YOLO class == 'bottle'
      – draws green box + name + MSRP / Shelf price
"""
import os, sys, cv2, numpy as np, torch, streamlit as st, traceback
from PIL import Image
import unicodedata
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# ─────────────── TURN / STUN config  ──────────────
RTC_CONFIGURATION = {
    "iceTransportPolicy": "relay",                 # force relay via TURN
    "iceServers": [
        {"urls": "stun:stun.l.google.com:19302"},  # public STUN
        {                                          # TURN relay (plain TCP 443)
            "urls": "turn:34.31.80.206:443?transport=tcp",
            "username": "streamlit",
            "credential": "SuperSecretPassword123",
        },
    ],
}

st.set_page_config(page_title="🥃 Whisky Goggles", layout="centered")

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

def clean_text(s):
    """Remove accents and unrenderable characters for OpenCV text."""
    s = unicodedata.normalize("NFKD", str(s))
    return s.encode("ascii", "ignore").decode("ascii")

def draw_text_with_bg(img, text, pos, font=cv2.FONT_HERSHEY_SIMPLEX,
                      font_scale=0.6, font_thickness=1,
                      text_color=(255,255,255), bg_color=(0,128,0), alpha=0.6):
    """Draw readable text with a semi-transparent background box."""
    x, y = pos
    text_size, baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size

    overlay = img.copy()
    cv2.rectangle(overlay, (x, y - text_h - baseline), (x + text_w, y + baseline),
                  bg_color, thickness=-1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    cv2.putText(img, text, (x, y), font, font_scale, text_color, font_thickness, lineType=cv2.LINE_AA)

def annotate_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """Detect bottles, match via CLIP, and draw name + price on BGR frame."""
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = DETECTOR.predict(img_rgb, verbose=False)[0]
    out = frame_bgr.copy()

    for xyxy, cls in zip(results.boxes.xyxy.cpu().numpy(),
                         results.boxes.cls.cpu().numpy().astype(int)):
        if cls != BOTTLE_CLS:
            continue

        x1, y1, x2, y2 = map(int, xyxy)
        crop = img_rgb[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        pil_crop = Image.fromarray(crop)
        vec = embed_pil(pil_crop)
        sims = FEATS_IMG @ vec
        best_idx = int(np.argmax(sims))
        bid = ID_LIST[best_idx]
        rec = META_DF.loc[bid]
        name, msrp, shelf = rec["name"], rec["avg_msrp"], rec["shelf_price"]

        # Clean name and build display labels
        clean_name = clean_text(name)
        price_label = f"Price: ${shelf:.0f}"

        # Draw bounding box
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        y_txt = y1 - 10 if y1 > 30 else y2 + 20

        draw_text_with_bg(out, clean_name, (x1, y_txt),
                          font_scale=0.7, font_thickness=2, bg_color=(40, 40, 40))

        draw_text_with_bg(out, price_label, (x1, y_txt + 22),
                          font_scale=0.55, font_thickness=1, bg_color=(40, 40, 40))
    return out

        
class BottleTransformer(VideoTransformerBase):
    def transform(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            annotated = annotate_frame(img)
            return annotated
        except Exception as e:
            print("⚠️ transform() error:", e)
            traceback.print_exc()
            return frame.to_ndarray(format="bgr24")  # fallback to raw frame

# ───────────────── UI LAYOUT ─────────────────

st.title("🥃 Whisky Goggles")

st.caption("Real-time whisky bottle identification using CLIP, YOLO, and PaddleOCR.")
st.caption("Achieves 100% top-1 accuracy on the 500-bottle catalog and 90%+ accuracy on unseen clear bottle images.")
st.caption("Supports live detection via webcam using WebRTC.")


tab_upload, tab_camera, tab_live = st.tabs(
    ["📁 Upload Photo","📸 Take Photo","🎥 Real Time"]
)

# --- Tab 1: Upload Photo ---
with tab_upload:
    st.subheader("Upload Bottle Photograph")
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
            

            st.write(f"**MSRP:** ${top.get('avg_msrp','N/A')}")
            st.write(f"**Shelf:** ${top.get('shelf_price','N/A')}")

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
                    f"{alt['rank']-1}. **{alt['name']}** "
                    f"({alt['confidence']*100:.2f}%)"
                )

# --- Tab 2: Take Photo ---
with tab_camera:
    st.subheader("Take a Photo")
    if "cam_on" not in st.session_state:
        st.session_state.cam_on = False

    col1, col2 = st.columns([1,3])
    if col1.button("▶️ Start Camera "):
        st.session_state.cam_on = True
    if col2.button("⏹️ Stop Camera "):
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

            c1, c2 = st.columns([3,2])
            with c1:
                st.markdown(f"### {top['name']}")
                st.markdown(f"**Confidence:** {top['confidence']*100:.2f}%")
                

                st.write(f"**MSRP:** ${top.get('avg_msrp','N/A')}")
                st.write(f"**Shelf:** ${top.get('shelf_price','N/A')}")

                status = {
                    "in_dataset":      "✅ Exact catalog match",
                    "unknown":         "⚠️ Closest match (not in catalog)",
                    "no_text_detected":"🔎 No label text detected"
                }[res["status"]]
                st.info(status)
            with c2:
                st.image(top["ref_img"], caption="Catalog reference")

            with st.expander("Other matches"):
                for alt in res["alt"]:
                    st.markdown(
                        f"{alt['rank']-1}. {alt['name']} "
                        f"({alt['confidence']*100:.2f}%)"
                    )
    else:
        st.info("Camera is off. Click ▶️ to start.")

# --- Tab 3: Real‑Time ---
with tab_live:
    st.subheader("Live Bottle Detector (WebRTC)")
    st.caption("Point camera in direction of the bottle to identify in real-time")
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
            rtc_configuration=RTC_CONFIGURATION,      # <─ NEW
            video_transformer_factory=BottleTransformer,
            async_processing=True,

            media_stream_constraints={
                "video": {
                    "width": 640,
                    "height": 480,
                    "frameRate": 15
                },
                "audio": False
            },
        )
    else:
        st.info("Camera is off. Click ▶️ to start real‑time detection.")

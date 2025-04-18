import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from scripts.match import match
from PIL import Image

st.set_page_config(page_title="🥃 Whisky Goggles", layout="centered")
st.title("🥃 Whisky Goggles")
st.caption("Hybrid CLIP+OCR recogniser over a 500‑bottle catalog")

# --- Input: camera or upload ---
cam   = st.camera_input("Take a photo")
upl   = st.file_uploader("…or upload", type=["jpg","png"])
imgf  = cam or upl

if imgf:
    tmp = "temp_query.jpg"
    with open(tmp, "wb") as f:
        f.write(imgf.read())

    st.image(Image.open(tmp), caption="Query", use_column_width=True)
    st.info("🔍 Matching…")
    res = match(tmp)

    top = res["top"]
    c1, c2 = st.columns([1,1])
    with c1:
        status = (
            "✅ Exact catalog match"
            if res["status"]=="in_dataset"
            else "⚠️ Closest match (not in catalog)"
        )
        st.subheader(status)
        st.markdown(f"**{top['name']}**  ({top['confidence']*100:.2f}% confidence)")
        st.markdown(f"MSRP ${top.get('avg_msrp','N/A')}  |  Shelf ${top.get('shelf_price','N/A')}")

    with c2:
        st.image(top["ref_img"], caption="Catalog reference")

    with st.expander("Other good matches"):
        for alt in res["alt"]:
            st.markdown(f"{alt['rank']}. **{alt['name']}** ({alt['confidence']*100:.2f}%)")

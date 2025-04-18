# app/streamlit_app.py
import sys
import os

# ensure project root is on PYTHONPATH if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from scripts.match import match
from PIL import Image

st.set_page_config(page_title="Whisky Goggles", layout="centered")
st.title("🥃 Whisky Goggles")
st.write("Upload a bottle photo and see the top matches with confidence scores.")

uploaded = st.file_uploader("Drop a photo", type=["jpg","png"])
if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Your Photo", use_container_width=True)

    tmp = "temp_query.jpg"
    img.save(tmp)
    results = match(tmp, top_k=5)

    for res in results:
        st.markdown(
            f"### {res['name']}  "
            f"(confidence: {res['confidence']:.4f})"
        )
        st.markdown(f"- **MSRP:** ${res['avg_msrp']}")
        st.markdown(f"- **Fair price:** ${res['fair_price']}")
        st.markdown(f"- **Shelf price:** ${res['shelf_price']}")
        st.write("---")
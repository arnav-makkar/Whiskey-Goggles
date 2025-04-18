import pandas as pd
import requests
import os
import logging

# --- Setup logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("scripts/download_images.log"),
        logging.StreamHandler()
    ]
)

df = pd.read_csv("data/bottles.csv", dtype={"id": str, "image_url": str})
os.makedirs("data/images", exist_ok=True)

for idx, row in df.iterrows():
    img_id = row["id"]
    url = row.get("image_url", "")
    
    # Skip missing URLs
    if not isinstance(url, str) or url.strip() == "":
        logging.warning(f"{img_id}: No image URL, skipping.")
        continue

    out_path = f"data/images/{img_id}.jpg"
    if os.path.exists(out_path):
        logging.info(f"{img_id}: Already downloaded, skipping.")
        continue

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        logging.error(f"{img_id}: Failed to download {url!r} → {e}")
        continue

    with open(out_path, "wb") as f:
        f.write(resp.content)
    logging.info(f"{img_id}: Saved {out_path}")

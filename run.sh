# pip install -r requirements.txt
python scripts/download_images.py
python scripts/extract_features.py
python scripts/match.py img.jpg
python scripts/match.py b2.jpg
streamlit run app.py
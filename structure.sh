whisky_goggles/
├── data/
│   ├── bottles.csv              # 500‑bottle metadata (id, name, image_url, pricing…)
│   └── images/                  # downloaded reference images (named {id}.jpg)
│
├── scripts/
│   ├── download_images.py       # grab all image_url → data/images/{id}.jpg
│   ├── extract_features.py      # compute and store reference embeddings
│   └── match.py                 # given a query image, return top‑k matches
│
├── app.py
│
├── requirements.txt
└── README.md

# 🥃 Whiskey Goggles

## Overview
Whiskey Goggles is a Streamlit-based web application designed to identify whiskey bottles; their name and price from input user images in real time. 

Link: https://bit.ly/whiskey-goggles

The web-app supports three modes of input:

1. **Upload Photo** – match a static image  
2. **Take Photo** – capture from your webcam  
3. **Real-Time** – live detection via WebRTC  

Behind the scenes, the system uses a hybrid two-stage matching pipeline:

- **Stage A:** ResNet-50 embeddings for exact matches from a catalog of 500 bottles  
- **Stage B:** YOLOv8s for detection, CLIP for vision-language embedding, and PaddleOCR for robust fallback on label text

---

## ✨ Features

- ✅ **100% top-1 accuracy** on catalog images using ResNet-2048 embeddings  
- 🔄 **Hybrid fallback system** combining YOLO, CLIP, and OCR for robust real-world predictions  
- 🧠 **Text + image embedding fusion** using CLIP and fuzzy string matching  
- 🌐 **Real-time detection** using browser WebRTC with TURN relay (TCP 443) for compatibility with Cloud Run  
- 🎯 **Clean UI overlays** showing name and shelf price with smooth semi-transparent labels  
- ☁️ **Cloud-native deployment** using Docker + Google Cloud Run + TURN server on GCE  

---

## ⚙️ Implementation Approach

### 1. Stage A: Exact Matching (ResNet-50)
- All catalog bottle images are pre-embedded using ResNet-50 and stored in `feats_norm.npy`
- For each query image, ResNet features are extracted and compared via cosine similarity
- If similarity ≥ 0.92, the system returns the best match with up to 3 closest alternatives

### 2. Stage B: Fallback Matching (YOLO + CLIP + OCR)
- If no exact match is found, YOLOv8s is used to detect bottles and crop the largest one
- The cropped image is embedded using CLIP and compared against image embeddings (`feats_img.npy`)
- If similarity ≥ 0.98, it is returned as a fallback match
- Otherwise, OCR is used to extract text from the crop, which is embedded and compared against text embeddings (`feats_txt.npy`)
- The final ranking combines visual similarity, text embedding similarity, and fuzzy string match

### 3. Streamlit Front-End
- The interface offers three tabs: Upload Photo, Take Photo, and Real-Time Detection
- The real-time camera feed continuously detects bottles and overlays name and shelf price live
- WebRTC is used for low-latency streaming, routed via a TURN server over TCP 443 to ensure Cloud Run compatibility

### 4. Deployment
- The application is containerized using Docker and deployed on Google Cloud Run
- A separate TURN server is hosted on a Google Compute Engine VM using `coturn` on port 443
- The backend loads models and embeddings efficiently using Streamlit caching for responsiveness

---

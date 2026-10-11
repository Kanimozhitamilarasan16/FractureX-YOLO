---
title: FractureX-YOLO
emoji: 🩻
colorFrom: blue
colorTo: gray
sdk: streamlit
sdk_version: 1.40.0
app_file: app.py
pinned: false
---

# FractureX-YOLO

Research demo for pediatric wrist-fracture detection using YOLO11s + custom DSC-C2f modules.

## Deploy to Hugging Face Spaces

1. Create a new Space at https://huggingface.co/new-space.
2. Select **Streamlit** as the SDK.
3. Upload `app.py`, `requirements.txt`, and this `README.md`.
4. Upload your trained checkpoint named **`best.pt`** to the Space repository root (same level as `app.py`). The checkpoint is not included in this source ZIP.
5. Wait for the build to finish, then test the Space with a wrist X-ray.

The app preserves the notebook's grayscale conversion, optional CLAHE preprocessing, 640-pixel inference size, confidence threshold, NMS IoU threshold, and bounding-box output.

**Important:** custom checkpoints may require the same compatible Ultralytics/PyTorch environment as training. If model loading fails, inspect the Space build/runtime logs and pin the training environment's versions.

**Medical disclaimer:** This is an experimental research prototype, not a medical device. Do not use it for diagnosis or treatment decisions.

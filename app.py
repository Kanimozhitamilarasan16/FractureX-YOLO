"""FractureX-YOLO Streamlit research demo.

Upload a wrist X-ray to run the custom YOLO11s + DSC-C2f detector.
Research prototype only; not a clinical diagnostic tool.
"""

import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from ultralytics import YOLO
from ultralytics.nn.modules import Conv, C2f

# -------------------------------------------------------------------
# Custom DSC-C2f classes copied from the provided training notebook.
# Keep these definitions available before YOLO loads best.pt because
# the serialized checkpoint contains references to these custom types.
# -------------------------------------------------------------------

class DSConv(nn.Module):
    """Dynamic snake convolution along one axis."""
    def __init__(self, c1, c2, morph, k=3, extend=1.0):
        super().__init__()
        self.k, self.morph, self.extend = k, morph, extend
        self.offset = nn.Sequential(nn.Conv2d(c1, k, 3, padding=1), nn.BatchNorm2d(k))
        self.proj = nn.Conv2d(c1 * k, c2, 1, bias=False)
        self.bn, self.act = nn.BatchNorm2d(c2), nn.SiLU()

    def forward(self, x):
        b, _, h, w = x.shape
        k, c = self.k, self.k // 2
        off = torch.tanh(self.offset(x)) * self.extend
        right = torch.cumsum(off[:, c + 1:], 1)
        left = torch.flip(torch.cumsum(torch.flip(off[:, :c], [1]), 1), [1])
        cum = torch.cat([left, torch.zeros_like(off[:, :1]), right], 1).float()
        ys, xs = torch.meshgrid(
            torch.arange(h, device=x.device, dtype=torch.float32),
            torch.arange(w, device=x.device, dtype=torch.float32),
            indexing="ij",
        )
        xf, outs = x.float(), []
        for i in range(k):
            s = i - c
            if self.morph == 0:
                gx = (xs + s).unsqueeze(0).expand(b, -1, -1)
                gy = ys + cum[:, i]
            else:
                gx = xs + cum[:, i]
                gy = (ys + s).unsqueeze(0).expand(b, -1, -1)
            grid = torch.stack(
                (2 * gx / max(w - 1, 1) - 1, 2 * gy / max(h - 1, 1) - 1), -1
            )
            outs.append(F.grid_sample(
                xf, grid, mode="bilinear", padding_mode="zeros", align_corners=True
            ))
        y = torch.cat(outs, 1).to(x.dtype)
        return self.act(self.bn(self.proj(y)))


class DySnakeConv(nn.Module):
    def __init__(self, c1, c2, k=3):
        super().__init__()
        self.c0 = Conv(c1, c2, k)
        self.cx, self.cy = DSConv(c1, c2, 0, k), DSConv(c1, c2, 1, k)
        self.fuse = Conv(3 * c2, c2, 1)

    def forward(self, x):
        return self.fuse(torch.cat([self.c0(x), self.cx(x), self.cy(x)], 1))


class Bottleneck_DSC(nn.Module):
    def __init__(self, c1, c2, shortcut=True):
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 1)
        self.cv2 = DySnakeConv(c2, c2, 3)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f_DSC(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Bottleneck_DSC(self.c, self.c, shortcut) for _ in range(n))




@st.cache_resource
def load_model():
    model_path = Path(os.getenv("MODEL_PATH", "best.pt"))
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {model_path.resolve()}. Upload best.pt beside app.py."
        )
    return YOLO(str(model_path))

st.set_page_config(page_title="FractureX-YOLO", page_icon="🩻", layout="wide")
st.title("🩻 FractureX-YOLO")
st.subheader("YOLO11s + DSC-C2f + Focaler-CIoU")
st.write(
    "Upload a wrist X-ray to view predicted fracture regions, confidence scores, "
    "and bounding-box coordinates. This is a research prototype, not a clinical diagnostic tool."
)

with st.sidebar:
    st.header("Detection settings")
    confidence = st.slider("Confidence threshold", 0.05, 0.90, 0.25, 0.05)
    iou = st.slider("NMS IoU threshold", 0.10, 0.90, 0.70, 0.05)
    use_clahe = st.checkbox("Apply CLAHE preprocessing", value=True)
    st.caption("Image size: 640 × 640 (model inference setting)")

uploaded = st.file_uploader("Upload wrist X-ray", type=["png", "jpg", "jpeg", "bmp", "webp"])
if uploaded is not None:
    pil_image = Image.open(uploaded).convert("RGB")
    image_array = np.asarray(pil_image)
    left, right = st.columns(2)
    with left:
        st.markdown("#### Original X-ray")
        st.image(pil_image, use_container_width=True)
    run = st.button("🔍 Detect fracture", type="primary", use_container_width=True)
    if run:
        try:
            with st.spinner("Loading model and analyzing X-ray..."):
                model = load_model()
                # Use the same preprocessing and post-processing as the notebook.
                start_time = time.time()
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
                if use_clahe:
                    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
                bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                result = model.predict(
                    source=bgr, imgsz=640, conf=float(confidence), iou=float(iou),
                    device=device, verbose=False
                )[0]
                annotated = cv2.cvtColor(result.plot(line_width=2), cv2.COLOR_BGR2RGB)
                elapsed = time.time() - start_time

            with right:
                st.markdown("#### Detection result")
                st.image(annotated, use_container_width=True)

            rows = []
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.detach().cpu().numpy()
                scores = result.boxes.conf.detach().cpu().numpy()
                for idx, (box, score) in enumerate(zip(boxes, scores), start=1):
                    rows.append({
                        "Region": idx,
                        "Confidence": round(float(score), 4),
                        "x1": int(box[0]), "y1": int(box[1]),
                        "x2": int(box[2]), "y2": int(box[3]),
                    })

            st.divider()
            c1, c2, c3 = st.columns(3)
            if rows:
                best_conf = max(row["Confidence"] for row in rows)
                c1.metric("Prediction", "FRACTURE DETECTED")
                c2.metric("Best confidence", f"{best_conf:.2%}")
                c3.metric("Detected regions", str(len(rows)))
                st.success(f"{len(rows)} region(s) detected above the {confidence:.0%} confidence threshold.")
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                c1.metric("Prediction", "NO FRACTURE DETECTED")
                c2.metric("Best confidence", "—")
                c3.metric("Detected regions", "0")
                st.warning(
                    f"No region detected above the {confidence:.0%} confidence threshold. "
                    "This does not rule out a fracture."
                )
            st.caption(f"Inference time: {elapsed:.3f} s")
        except Exception as exc:
            st.error("The model could not run. Check that best.pt matches the custom DSC-C2f classes and installed Ultralytics version.")
            st.exception(exc)
else:
    st.info("Choose a wrist X-ray image above to get started.")

st.divider()
st.warning("Research demonstration only — not intended for clinical diagnosis or treatment decisions.")

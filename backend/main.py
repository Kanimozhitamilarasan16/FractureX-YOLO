"""FractureX-YOLO FastAPI inference service; research use only."""
import __main__
import os
from functools import lru_cache
from pathlib import Path
from io import BytesIO
import traceback

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO
from ultralytics.nn.modules import Conv, C2f


# These custom classes must exist before loading the serialized checkpoint.
class DSConv(nn.Module):
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
            torch.arange(w, device=x.device, dtype=torch.float32), indexing="ij"
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
            grid = torch.stack((2 * gx / max(w - 1, 1) - 1,
                                2 * gy / max(h - 1, 1) - 1), -1)
            outs.append(F.grid_sample(xf, grid, mode="bilinear",
                                      padding_mode="zeros", align_corners=True))
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


app = FastAPI(title="FractureX-YOLO API", version="1.0.0",
              description="Research prototype for pediatric wrist X-ray detection.")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=False,
    allow_methods=["GET", "POST"], allow_headers=["*"]
)


def checkpoint_path():
    return Path(os.getenv("MODEL_PATH", str(Path(__file__).resolve().parent.parent / "best.pt")))



@lru_cache(maxsize=1)
def load_model():
    path = checkpoint_path()

    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    __main__.DSConv = DSConv
    __main__.DySnakeConv = DySnakeConv
    __main__.Bottleneck_DSC = Bottleneck_DSC
    __main__.C2f_DSC = C2f_DSC

    model = YOLO(str(path))
    model.to("cpu")
    return model


@app.get("/")
def root():
    return {"name": "FractureX-YOLO API", "status": "running",
            "health": "/health", "predict": "POST /predict",
            "disclaimer": "Research prototype only; not for clinical diagnosis."}


@app.get("/health")
def health():
    return {"status": "ok", "checkpoint_found": checkpoint_path().is_file(), "device": "cpu"}


@app.post("/predict")
async def predict(file: UploadFile = File(...),
                  confidence: float = Query(0.25, ge=0.05, le=0.90),
                  iou: float = Query(0.70, ge=0.10, le=0.90),
                  use_clahe: bool = Query(True)):
    if file.content_type not in {"image/png", "image/jpeg", "image/webp", "image/bmp"}:
        raise HTTPException(415, "Upload PNG, JPG, JPEG, WEBP, or BMP.")
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Uploaded file is empty.")
    if len(raw) > 12 * 1024 * 1024:
        raise HTTPException(413, "Image must be 12 MB or smaller.")
    try:
        rgb = np.asarray(Image.open(BytesIO(raw)).convert("RGB"))
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Invalid image file.")

    try:
        model = load_model()
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if use_clahe:
            gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        with torch.inference_mode():
            result = model.predict(source=bgr, imgsz=640, conf=confidence,
                                   iou=iou, device="cpu", verbose=False)[0]
        detections = []
        if result.boxes is not None and len(result.boxes):
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            scores = result.boxes.conf.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            for n, (box, score, cls_id) in enumerate(zip(boxes, scores, classes), 1):
                detections.append({
                    "region": n, "class_id": int(cls_id),
                    "class_name": str(result.names.get(int(cls_id), cls_id)),
                    "confidence": round(float(score), 5),
                    "x1": round(float(box[0]), 2), "y1": round(float(box[1]), 2),
                    "x2": round(float(box[2]), 2), "y2": round(float(box[3]), 2)
                })
        return {
            "filename": file.filename,
            "prediction": "DETECTION FOUND" if detections else "NO DETECTION ABOVE THRESHOLD",
            "detection_count": len(detections), "confidence_threshold": confidence,
            "iou_threshold": iou, "image_width": int(rgb.shape[1]),
            "image_height": int(rgb.shape[0]), "detections": detections,
            "disclaimer": "Research prototype only. No detection does not rule out fracture; not for clinical diagnosis."
        }
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        print(f"INFERENCE ERROR: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed ({type(exc).__name__}): {exc}"
        )   

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))

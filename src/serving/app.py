from __future__ import annotations

import base64
import io
from typing import Any, Dict, List

from fastapi import Body, FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from .schemas import Health
from .inference import get_det, get_seg, get_cls

app = FastAPI(title="CV API", version="1.0", docs_url="/docs")


def _file_to_image(file: UploadFile) -> Image.Image:
    return Image.open(io.BytesIO(file.file.read())).convert("RGB")


def _b64_to_image(s: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")


@app.get("/health")
def health():
    # You can enrich this with which models are loaded by calling the getters
    return Health(
        status="ok",
        model_backend="yolo",
        model_version="multi-task",
    ).model_dump()


# detection
@app.post("/predict", tags=["detection"])
def detect(file: UploadFile = File(...)):
    img = _file_to_image(file)
    pred = get_det().predict(img)
    return JSONResponse(pred)


@app.post("/v1/models:predict", tags=["detection"])
def detect_vertex(payload: Dict[str, Any] = Body(...)):
    inst = payload.get("instances") or []
    preds: List[Dict[str, Any]] = []
    for it in inst:
        b64 = it.get("b64") if isinstance(it, dict) else it
        img = _b64_to_image(b64)
        preds.append(get_det().predict(img))
    return JSONResponse({"predictions": preds})


# segmentation
@app.post("/segment", tags=["segmentation"])
def segment(file: UploadFile = File(...)):
    img = _file_to_image(file)
    pred = get_seg().predict(img)
    return JSONResponse(pred)


@app.post("/v1/segment:predict", tags=["segmentation"])
def segment_vertex(payload: Dict[str, Any] = Body(...)):
    inst = payload.get("instances") or []
    preds: List[Dict[str, Any]] = []
    for it in inst:
        b64 = it.get("b64") if isinstance(it, dict) else it
        img = _b64_to_image(b64)
        preds.append(get_seg().predict(img))
    return JSONResponse({"predictions": preds})


# classification
@app.post("/classify", tags=["classification"])
def classify(file: UploadFile = File(...)):
    img = _file_to_image(file)
    pred = get_cls().predict(img)
    return JSONResponse(pred)


@app.post("/v1/classify:predict", tags=["classification"])
def classify_vertex(payload: Dict[str, Any] = Body(...)):
    inst = payload.get("instances") or []
    preds: List[Dict[str, Any]] = []
    for it in inst:
        b64 = it.get("b64") if isinstance(it, dict) else it
        img = _b64_to_image(b64)
        preds.append(get_cls().predict(img))
    return JSONResponse({"predictions": preds})

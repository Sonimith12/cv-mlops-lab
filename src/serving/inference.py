import os
from functools import lru_cache
from typing import Any, Dict, List

from PIL import Image


class BaseService:
    model_version: str = "unknown"

    def predict(self, img: Image.Image) -> Dict[str, Any]:
        raise NotImplementedError


class YOLODetService(BaseService):
    """
    Returns:
      {
        "bboxes": [
            {"x1":..., "y1":..., "x2":..., "y2":..., "conf":..., "cls":"..."},
            ...
        ],
        "model_version": "path-or-name"
      }
    """

    def __init__(
        self, model_path: str | None = None, conf: float = 0.35, iou: float = 0.45
    ):
        from ultralytics import YOLO

        mp = (
            model_path
            or os.getenv("DET_MODEL_PATH")
            or os.getenv("MODEL_PATH")
            or "yolov8n.pt"
        )
        self.model = YOLO(mp)
        self.model_version = str(mp)
        self.conf = conf
        self.iou = iou

    def predict(self, img: Image.Image) -> Dict[str, Any]:
        r = self.model.predict(img, conf=self.conf, iou=self.iou, verbose=False)[0]
        names = r.names
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return {"bboxes": [], "model_version": self.model_version}

        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy()
        conf = boxes.conf.cpu().numpy()

        bboxes: List[Dict[str, Any]] = []
        for (x1, y1, x2, y2), k, c in zip(xyxy, cls, conf):
            bboxes.append(
                {
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "conf": float(c),
                    "cls": str(names[int(k)]),
                }
            )

        return {"bboxes": bboxes, "model_version": self.model_version}


class YOLOSegService(BaseService):
    """
    Returns:
      {
        "masks": [
            {"points": [[x,y],...], "cls":"...", "conf":0.9},
            ...
        ],
        "model_version": "path-or-name"
      }
    """

    def __init__(
        self, model_path: str | None = None, conf: float = 0.35, iou: float = 0.45
    ):
        from ultralytics import YOLO

        mp = model_path or os.getenv("SEG_MODEL_PATH") or "yolov8n-seg.pt"
        self.model = YOLO(mp)
        self.model_version = str(mp)
        self.conf = conf
        self.iou = iou

    def predict(self, img: Image.Image) -> Dict[str, Any]:
        r = self.model.predict(img, conf=self.conf, iou=self.iou, verbose=False)[0]
        names = r.names
        masks_out: List[Dict[str, Any]] = []

        # r.masks.xy is a list of Nx2 numpy arrays (polygon per instance)
        if getattr(r, "masks", None) is not None and r.masks is not None:
            polys = r.masks.xy
            cls = r.boxes.cls.cpu().numpy() if r.boxes is not None else []
            conf = r.boxes.conf.cpu().numpy() if r.boxes is not None else []

            for poly, k, c in zip(polys, cls, conf):
                pts = [[float(x), float(y)] for x, y in poly]
                masks_out.append(
                    {"points": pts, "cls": str(names[int(k)]), "conf": float(c)}
                )

        return {"masks": masks_out, "model_version": self.model_version}


class YOLOClsService(BaseService):
    """
    Returns:
      {
        "topk": [["label", prob], ...],  # prob in [0,1]
        "model_version": "path-or-name"
      }
    """

    def __init__(self, model_path: str | None = None, topk: int = 5):
        from ultralytics import YOLO

        mp = model_path or os.getenv("CLS_MODEL_PATH") or "yolov8n-cls.pt"
        self.model = YOLO(mp)
        self.model_version = str(mp)
        self.topk = int(os.getenv("CLS_TOPK", topk))

    def predict(self, img: Image.Image) -> Dict[str, Any]:
        r = self.model.predict(img, verbose=False)[0]
        names = r.names  # index -> label
        probs = getattr(r, "probs", None)

        if probs is None or probs.data is None:
            return {"topk": [], "model_version": self.model_version}

        arr = probs.data.cpu().numpy().tolist()  # list of probabilities by index
        pairs = sorted(
            ((names[i], float(p)) for i, p in enumerate(arr)),
            key=lambda x: x[1],
            reverse=True,
        )[: self.topk]

        return {"topk": pairs, "model_version": self.model_version}


@lru_cache(maxsize=1)
def get_det() -> YOLODetService:
    return YOLODetService()


@lru_cache(maxsize=1)
def get_seg() -> YOLOSegService:
    return YOLOSegService()


@lru_cache(maxsize=1)
def get_cls() -> YOLOClsService:
    return YOLOClsService()
